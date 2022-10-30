use std::cell::SyncUnsafeCell;

use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::inference::{CHANNEL_COUNT, POLICY_LEN, featurize_state};
use crate::rules::{Move, State};

pub const BATCH_SIZE: usize = 128;

// We double buffer, to improve performance.
pub const BUFFER_COUNT: usize = 2;

struct ReturnChannels {
  next_buffer: usize,
  slot_index:  usize,
  channels:    [Option<tokio::sync::oneshot::Sender<ModelOutputs>>; BUFFER_COUNT * BATCH_SIZE],
}

pub struct TensorFlowEngine {
  bundle:          &'static SavedModelBundle,
  input_op:        Operation,
  policy_op:       Operation,
  value_op:        Operation,
  input_tensors:   [&'static SyncUnsafeCell<Tensor<f32>>; BUFFER_COUNT],
  return_channels: tokio::sync::Mutex<ReturnChannels>,
  eval_count:      &'static std::sync::atomic::AtomicUsize,
}

impl TensorFlowEngine {
  pub async fn create(model_dir: &str) -> TensorFlowEngine {
    // Initialize model_dir, input tensor, and an empty graph
    let input_tensors = (0..BUFFER_COUNT)
      .map(|_| {
        &*Box::leak(Box::new(SyncUnsafeCell::new(Tensor::new(&[
          BATCH_SIZE as u64,
          CHANNEL_COUNT as u64,
          8,
          8,
        ]))))
      })
      .collect::<Vec<_>>();
    let mut graph = Graph::new();

    println!("Loading model from {}", model_dir);

    // Load saved model bundle (session state + meta_graph data)
    let bundle = Box::leak(Box::new(
      SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)
        .expect("Can't load saved model"),
    ));

    // Get signature metadata from the model bundle
    let signature = bundle.meta_graph_def().get_signature("serving_default").unwrap();

    // Print all inputs.
    println!("Inputs:");
    for (name, tensor_info) in signature.inputs() {
      println!("  {}: {:?}", name, tensor_info);
    }
    // Print all outputs.
    println!("Outputs:");
    for (name, tensor_info) in signature.outputs() {
      println!("  {}: {:?}", name, tensor_info);
    }

    // Get input/output info.
    let input_info = signature.get_input("inp_features").unwrap();
    let policy_info = signature.get_output("out_policy").unwrap();
    let value_info = signature.get_output("out_value").unwrap();

    // Get input/output ops from graph.
    let input_op = graph.operation_by_name_required(&input_info.name().name).unwrap();
    let policy_op = graph.operation_by_name_required(&policy_info.name().name).unwrap();
    let value_op = graph.operation_by_name_required(&value_info.name().name).unwrap();

    let eval_count: &'static _ = Box::leak(Box::new(std::sync::atomic::AtomicUsize::new(0)));

    // Launch a thread to read eval_count periodically.
    tokio::spawn(async move {
      const SLEEP_INTERVAL: usize = 20;
      let mut last_eval_count = 0;
      loop {
        tokio::time::sleep(std::time::Duration::from_secs(SLEEP_INTERVAL as u64)).await;
        let eval_count = eval_count.load(std::sync::atomic::Ordering::Relaxed) * BATCH_SIZE;
        println!(
          "eval_count: {} ({} per second)",
          eval_count,
          (eval_count - last_eval_count) / SLEEP_INTERVAL
        );
        last_eval_count = eval_count;
      }
    });

    TensorFlowEngine {
      bundle,
      input_op,
      policy_op,
      value_op,
      input_tensors: input_tensors[..].try_into().unwrap(),
      return_channels: tokio::sync::Mutex::new(ReturnChannels {
        next_buffer: 0,
        slot_index:  0,
        channels:    (0..BUFFER_COUNT * BATCH_SIZE)
          .map(|_| None)
          .collect::<Vec<_>>()
          .try_into()
          .map_err(|_| ())
          .unwrap(),
      }),
      eval_count,
    }
  }
}

impl mcts::InferenceEngine for TensorFlowEngine {
  type Cookie = ();

  async fn predict(&self, cookie: Self::Cookie, state: &State) -> ModelOutputs {
    // Allocate a slot.
    let (rx, work) = {
      let mut guard = self.return_channels.lock().await;
      //println!("[{worker_id}] \x1b[93m >>> LOCKED\x1b[0m");
      let slot_index = guard.slot_index;
      guard.slot_index = (guard.slot_index + 1) % (BUFFER_COUNT * BATCH_SIZE);
      // Make sure we haven't overflowed.
      assert!(guard.channels[slot_index].is_none());
      // Featurize.
      let array: &mut [f32] = unsafe {
        let input_tensor: *mut Tensor<f32> = self.input_tensors[slot_index / BATCH_SIZE].get();
        const SLOT_SIZE: usize = 64 * CHANNEL_COUNT;
        let offset = (slot_index % BATCH_SIZE) * SLOT_SIZE;
        &mut input_tensor.as_mut().unwrap()[offset..offset + SLOT_SIZE]
      };
      featurize_state::<f32>(state, array.try_into().unwrap());

      //println!("[{worker_id}] Allocated slot {} (next_buffer={})", slot_index, guard.next_buffer);

      let work = match slot_index % BATCH_SIZE == BATCH_SIZE - 1 {
        true => {
          let buffer = guard.next_buffer;
          guard.next_buffer = (guard.next_buffer + 1) % BUFFER_COUNT;
          Some(buffer)
        }
        false => None,
      };

      // Insert a new channel.
      let (tx, rx) = tokio::sync::oneshot::channel();
      guard.channels[slot_index] = Some(tx);
      //println!("[{worker_id}] \x1b[93m <<< UNLOCKING\x1b[0m");
      (rx, work)
    };

    if let Some(buffer) = work {
      //println!("[{worker_id}] >>>>>>>>>> Running inference");
      // If we're the last slot, run the batch.
      let (policy_output, value_output) = {
        let session = &self.bundle.session;
        let mut session_run_args = SessionRunArgs::new();
        session_run_args.add_feed(&self.input_op, 0, unsafe {
          &*self.input_tensors[buffer].get()
        });
        let policy_ft = session_run_args.request_fetch(&self.policy_op, 0);
        let value_ft = session_run_args.request_fetch(&self.value_op, 1);
        session.run(&mut session_run_args).expect("Can't run session");
        let policy_output = session_run_args.fetch::<f32>(policy_ft).expect("Can't fetch output");
        let value_output = session_run_args.fetch::<f32>(value_ft).expect("Can't fetch output");
        (policy_output, value_output)
      };
      self.eval_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

      let mut guard = self.return_channels.lock().await;
      //let mut guard = self.return_channels.lock().await;
      //println!("[{worker_id}] \x1b[92m >>> LOCKING: {buffer}\x1b[0m");
      // Acquire the lock.
      let relevant_return_channels =
        &mut guard.channels[buffer * BATCH_SIZE..(buffer + 1) * BATCH_SIZE];

      for (i, channel) in relevant_return_channels.iter_mut().enumerate() {
        let tx = channel.take().unwrap();
        tx.send(ModelOutputs {
          policy: policy_output[i * POLICY_LEN..(i + 1) * POLICY_LEN].try_into().unwrap(),
          value:  value_output[i],
        })
        .map_err(|_| ())
        .expect("failed to send");
      }
      //println!("[{worker_id}] \x1b[92m <<< UNLOCKING: {buffer}\x1b[0m");
      drop(guard);
    }

    //println!("[{worker_id}] \x1b[91mBlocking on inference:\x1b[0m {}", slot_index);
    let r = rx.await.unwrap();
    //println!("[{worker_id}] \x1b[94mUnblocked on inference:\x1b[0m {}", slot_index);
    r
  }
}

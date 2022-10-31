use std::cell::RefCell;
use std::sync::Mutex;

use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::inference::{self, PendingIndex};
use crate::inference::{
  featurize_state, InferenceResults, InputBlock, CHANNEL_COUNT, FEATURES_SIZE, POLICY_LEN,
};
use crate::mcts;
use crate::rules::{Move, State};

pub const MAX_BATCH_SIZE: usize = 128;

//struct ReturnChannels {
//  next_buffer: usize,
//  slot_index:  usize,
//  channels:    [Option<tokio::sync::oneshot::Sender<ModelOutputs>>; BUFFER_COUNT * BATCH_SIZE],
//}

pub struct TensorFlowEngine {
  bundle:       SavedModelBundle,
  input_op:     Operation,
  policy_op:    Operation,
  value_op:     Operation,
  input_blocks: Mutex<Vec<InputBlock<MAX_BATCH_SIZE>>>,
  //input_tensors:   [&'static SyncUnsafeCell<Tensor<f32>>; BUFFER_COUNT],
  //return_channels: tokio::sync::Mutex<ReturnChannels>,
  eval_count:   std::sync::atomic::AtomicUsize,
}

impl TensorFlowEngine {
  pub fn new(model_dir: &str) -> TensorFlowEngine {
    //// Initialize model_dir, input tensor, and an empty graph
    //let input_tensors = (0..BUFFER_COUNT)
    //  .map(|_| {
    //    &*Box::leak(Box::new(SyncUnsafeCell::new(Tensor::new(&[
    //      BATCH_SIZE as u64,
    //      CHANNEL_COUNT as u64,
    //      8,
    //      8,
    //    ]))))
    //  })
    //  .collect::<Vec<_>>();
    let mut graph = Graph::new();

    println!("Loading model from {}", model_dir);

    // Load saved model bundle (session state + meta_graph data)
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)
      .expect("Can't load saved model");

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

    //let eval_count: &'static _ = Box::leak(Box::new(std::sync::atomic::AtomicUsize::new(0)));

    TensorFlowEngine {
      bundle,
      input_op,
      policy_op,
      value_op,
      //input_tensors: input_tensors[..].try_into().unwrap(),
      //return_channels: tokio::sync::Mutex::new(ReturnChannels {
      //  next_buffer: 0,
      //  slot_index:  0,
      //  channels:    (0..BUFFER_COUNT * BATCH_SIZE)
      //    .map(|_| None)
      //    .collect::<Vec<_>>()
      //    .try_into()
      //    .map_err(|_| ())
      //    .unwrap(),
      //}),
      input_blocks: Mutex::new(vec![]),
      eval_count: std::sync::atomic::AtomicUsize::new(0),
    }
  }
}

impl inference::InferenceEngine for TensorFlowEngine {
  const DESIRED_BATCH_SIZE: usize = MAX_BATCH_SIZE;

  fn add_work(&self, state: &crate::rules::State, cookie: PendingIndex) -> usize {
    println!("\x1b[93mAdding work...\x1b[0m");
    let mut input_blocks = self.input_blocks.lock().unwrap();
    inference::add_to_input_blocks(&mut input_blocks, state, cookie)
    /*
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
    */
  }

  fn predict(&self, use_outputs: impl FnOnce(InferenceResults)) -> usize {
    println!("\x1b[93mPredicting...\x1b[0m");
    // Pop the last input block, which is inside the mutex.
    let last_block = {
      let mut guard = self.input_blocks.lock().unwrap();
      match guard.pop() {
        Some(block) => block,
        None => return 0,
      }
    };
    let block_len = last_block.cookies.len();
    assert!(block_len <= MAX_BATCH_SIZE);
    // Create the input tensor.
    let input_data = &last_block.data[..block_len * FEATURES_SIZE];
    let input_tensor = Tensor::new(&[block_len as u64, CHANNEL_COUNT as u64, 8, 8])
      .with_values(input_data)
      .unwrap();
    // Configure inputs and outputs.
    let mut session_run_args = SessionRunArgs::new();
    session_run_args.add_feed(&self.input_op, 0, &input_tensor);
    let policy_ft = session_run_args.request_fetch(&self.policy_op, 0);
    let value_ft = session_run_args.request_fetch(&self.value_op, 1);
    // Run the actual network.
    self.bundle.session.run(&mut session_run_args).expect("Can't run session");
    self.eval_count.fetch_add(block_len, std::sync::atomic::Ordering::Relaxed);
    // Fetch outputs.
    let policy_output = session_run_args.fetch::<f32>(policy_ft).expect("Can't fetch output");
    let value_output = session_run_args.fetch::<f32>(value_ft).expect("Can't fetch output");
    let mut policies: Vec<&[f32; POLICY_LEN]> = vec![];
    for i in 0..block_len {
      let range = i * POLICY_LEN..(i + 1) * POLICY_LEN;
      let slice: &[f32] = &policy_output[range];
      // Policy should already be normalized.
      let sum: f32 = slice.iter().sum();
      debug_assert!((sum - 1.0).abs() < 1e-4);
      policies.push(slice.try_into().unwrap());
    }
    // Pass the outputs to the callback.
    use_outputs(InferenceResults::new(
      &last_block.cookies,
      &policies,
      &value_output[..],
    ));
    block_len

    /*
    if let Some(buffer) = work {
      //println!("[{worker_id}] >>>>>>>>>> Running inference");
      // If we're the last slot, run the batch.
      let (policy_output, value_output) = {

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
     */
  }

  fn clear(&self) {
    self.input_blocks.lock().unwrap().clear();
  }
}

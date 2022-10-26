use std::cell::{SyncUnsafeCell, Cell};

use tensorflow::{
  FetchToken, Graph, Operation, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor,
};

use crate::rules::{State, Move};

pub const BATCH_SIZE: usize = 256;

// We triple buffer, to improve performance.
pub const BUFFER_COUNT: usize = 2;

// We have:
//   Six channels for white pieces: pawns, knights, bishops, rooks, queens, kings
//   Six channels for black pieces.
//   One channel for ducks.
//   One channel for whose turn it is.
//   One channel for if it's the duck subturn.
//   Four channels for castling rights.
//   One channel for an en passant square.
//   One channel for the last move.
//   One channel of all ones.
const CHANNEL_COUNT: usize = 6 + 6 + 1 + 1 + 1 + 4 + 1 + 1 + 1;
const SLOT_SIZE: usize = 64 * CHANNEL_COUNT;

pub struct ModelOutputs {
  // policy[64 * from + to] is a probability 0 to 1.
  pub policy: [f32; 64 * 64],
  // value is a valuation for the current player from -1 to +1.
  pub value:  f32,
}

impl ModelOutputs {
  pub fn renormalize(&mut self, moves: &[Move]) {
    let mut temp = [0.0; 64 * 64];
    let mut sum = 0.0;
    for m in moves {
      let idx = (m.from % 64) as usize * 64 + m.to as usize;
      let val = self.policy[idx];
      temp[idx] = val;
      sum += val;
    }
    let rescale = 1.0 / (1e-16 + sum);
    for i in 0..64 * 64 {
      self.policy[i] = temp[i] * rescale;
    }
  }
}

fn featurize_state<T: From<u8>>(state: &State, array: &mut [T; 64 * CHANNEL_COUNT]) {
  let mut layer_index = 0;
  let mut emit_bitboard = |bitboard: u64| {
    for i in 0..64 {
      //array[64 * layer_index + i] = (((bitboard >> i) & 1) as u8).into();
      array[22 * i + layer_index] = (((bitboard >> i) & 1) as u8).into();
    }
    layer_index += 1;
  };
  let bool_board = |b: bool| if b { u64::MAX } else { 0 };
  // Encode the pieces.
  for player in [1, 0] {
    for piece_array in [
      &state.pawns,
      &state.knights,
      &state.bishops,
      &state.rooks,
      &state.queens,
      &state.kings,
    ] {
      emit_bitboard(piece_array[player].0);
    }
  }
  // Encode the ducks.
  emit_bitboard(state.ducks.0);
  // Encode whose turn it is.
  emit_bitboard(bool_board(state.white_turn));
  // Encode if it's the duck subturn.
  emit_bitboard(bool_board(state.is_duck_move));
  // Encode castling rights.
  for player in [1, 0] {
    emit_bitboard(bool_board(state.castling_rights[player].king_side));
    emit_bitboard(bool_board(state.castling_rights[player].queen_side));
  }
  // Encode en passant square.
  emit_bitboard(state.en_passant.0);
  // Encode last move.
  emit_bitboard(state.highlight.0);
  // Encode all ones.
  emit_bitboard(u64::MAX);
  assert_eq!(layer_index, CHANNEL_COUNT);
}

struct ReturnChannels {
  next_buffer: usize,
  slot_index:  usize,
  channels:    [Option<tokio::sync::oneshot::Sender<ModelOutputs>>; BUFFER_COUNT * BATCH_SIZE],
}

pub struct InferenceEngine {
  bundle:          &'static SavedModelBundle,
  input_op:        Operation,
  policy_op:       Operation,
  value_op:        Operation,
  //session_run_args: RefCell<SessionRunArgs<'a>>,
  //policy_ft: FetchToken,
  //value_ft: FetchToken,
  input_tensors:   [&'static SyncUnsafeCell<Tensor<f32>>; BUFFER_COUNT],
  return_channels: tokio::sync::Mutex<ReturnChannels>,
  eval_count:      &'static std::sync::atomic::AtomicUsize,
  //input_op: String,
  //output_op: String,
}

//unsafe impl Sync for InferenceEngine {}

impl InferenceEngine {
  pub async fn create() -> InferenceEngine {
    // Initialize save_dir, input tensor, and an empty graph
    let save_dir = "/tmp/keras";
    let input_tensors = (0..BUFFER_COUNT).map(|_| &*Box::leak(Box::new(
      SyncUnsafeCell::new(
      Tensor::new(&[BATCH_SIZE as u64, 8, 8, 22])
      )
    ))).collect::<Vec<_>>();
    let mut graph = Graph::new();

    // Load saved model bundle (session state + meta_graph data)
    let bundle = Box::leak(Box::new(
      SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, save_dir)
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
      let mut last_eval_count = 0;
      loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let eval_count = eval_count.load(std::sync::atomic::Ordering::Relaxed) * BATCH_SIZE;
        println!("eval_count: {} ({} per second)", eval_count, eval_count - last_eval_count);
        last_eval_count = eval_count;
      }
    });

    InferenceEngine {
      bundle,
      input_op,
      policy_op,
      value_op,
      input_tensors: input_tensors[..].try_into().unwrap(),
      return_channels: tokio::sync::Mutex::new(ReturnChannels {
        next_buffer: 0,
        slot_index: 0,
        channels: (0..BUFFER_COUNT*BATCH_SIZE).map(|_| None).collect::<Vec<_>>().try_into().map_err(|_| ()).unwrap(),
      }),
      eval_count,
    }
  }

  pub async fn predict(&self, worker_id: usize, state: &State) -> ModelOutputs {
    // Allocate a slot.
    let (slot_index, rx, work) = {
      let mut guard = self.return_channels.lock().await;
      //println!("[{worker_id}] \x1b[93m >>> LOCKED\x1b[0m");
      let slot_index = guard.slot_index;
      guard.slot_index = (guard.slot_index + 1) % (BUFFER_COUNT * BATCH_SIZE);
      // Make sure we haven't overflowed.
      let g: String = guard.channels.iter().map(|c| match c {
        Some(_) => "[ ]",
        None => " - ",
      }).collect();

      //println!("[{worker_id}] All channels: {:?} (index={})", g, slot_index);
      if guard.channels[slot_index].is_some() {
        //println!("[{worker_id}] \x1b[91m >>> OVERFLOW\x1b[0m");
        panic!("Overflowed!");
      }
      //assert!(guard.channels[slot_index].is_none());
      // Featurize.
      let array: &mut [f32] = unsafe {
        let input_tensor: *mut Tensor<f32> = self.input_tensors[slot_index / BATCH_SIZE].get();
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
      drop(guard);
      (slot_index, rx, work)
    };

    //let tensor_index = slot_index / BATCH_SIZE;
    ////println!("[{worker_id}] Allocated slot {} (tensor={})", slot_index, slot_index / BATCH_SIZE);

    if let Some(buffer) = work {

      //println!("[{worker_id}] >>>>>>>>>> Running inference");
      // If we're the last slot, run the batch.
      let (policy_output, value_output) = {
        let session = &self.bundle.session;
        let mut session_run_args = SessionRunArgs::new();
        session_run_args.add_feed(&self.input_op, 0, unsafe { &*self.input_tensors[buffer].get() });
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
      let relevant_return_channels = &mut guard.channels[buffer * BATCH_SIZE .. (buffer + 1) * BATCH_SIZE];

      for (i, channel) in relevant_return_channels.iter_mut().enumerate() {
        let tx = channel.take().unwrap();
        tx.send(ModelOutputs {
          policy: policy_output[i..i + 64*64].try_into().unwrap(),
          value: value_output[i],
        }).map_err(|_| ()).expect("failed to send");
      }
      let g: String = guard.channels.iter().map(|c| match c {
        Some(_) => "[ ]",
        None => " - ",
      }).collect();

      //println!("[{worker_id}] Channels after infr: {:?}", g);
      //println!("[{worker_id}] \x1b[92m <<< UNLOCKING: {buffer}\x1b[0m");
      drop(guard);
    }

    //println!("[{worker_id}] \x1b[91mBlocking on inference:\x1b[0m {}", slot_index);
    let r = rx.await.unwrap();
    //println!("[{worker_id}] \x1b[94mUnblocked on inference:\x1b[0m {}", slot_index);
    r
  }
}

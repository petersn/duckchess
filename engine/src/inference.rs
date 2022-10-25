use std::cell::RefCell;

use tensorflow::{
  FetchToken, Graph, Operation, SavedModelBundle, Session, SessionOptions, SessionRunArgs, Tensor,
};

use crate::rules::State;

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

fn featurize_state<T: From<u8>>(state: &State, array: &mut [T; 64 * CHANNEL_COUNT]) {
  let mut layer_index = 0;
  let mut emit_bitboard = |bitboard: u64| {
    for i in 0..64 {
      array[64 * layer_index + i] = (((bitboard >> i) & 1) as u8).into();
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

pub struct InferenceEngine {
  graph:        Graph,
  bundle:       &'static SavedModelBundle,
  input_op:     Operation,
  policy_op:    Operation,
  value_op:     Operation,
  //session_run_args: RefCell<SessionRunArgs<'a>>,
  //policy_ft: FetchToken,
  //value_ft: FetchToken,
  input_tensor: &'static RefCell<Tensor<f32>>,
  //input_op: String,
  //output_op: String,
}

impl InferenceEngine {
  pub async fn create() -> InferenceEngine {
    // Initialize save_dir, input tensor, and an empty graph
    let save_dir = "/tmp/keras";
    let input_tensor = Box::leak(Box::new(RefCell::new(
      Tensor::new(&[1, 8, 8, 22])
        .with_values(&[0.0; 1 * 8 * 8 * 22])
        .expect("Can't create tensor"),
    )));
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

    InferenceEngine {
      graph,
      bundle,
      input_op,
      policy_op,
      value_op,
      //session_run_args: RefCell::new(session_run_args),
      //policy_ft,
      //value_ft,
      input_tensor,
      //input_op: input_info.name().name.clone(),
      //output_op: policy_info.name().name.clone(),
    }
  }

  pub async fn predict(&self, state: &State, policy: &mut [f32; 64 * 64], value: &mut f32) {
    // Featurize.
    let mut input_tensor = self.input_tensor.borrow_mut();
    let array: &mut [f32] = input_tensor.as_mut();
    // Zero the array.
    for i in 0..array.len() {
      array[i] = 0.0;
    }
    //featurize_state::<f32>(state, array.try_into().unwrap());
    let session = &self.bundle.session;
    let mut session_run_args = SessionRunArgs::new();
    session_run_args.add_feed(&self.input_op, 0, &input_tensor);
    let policy_ft = session_run_args.request_fetch(&self.policy_op, 0);
    let value_ft = session_run_args.request_fetch(&self.value_op, 0);
    session.run(&mut session_run_args).expect("Can't run session");
    let policy_output = session_run_args.fetch::<f32>(policy_ft).expect("Can't fetch output");
    let value_output = session_run_args.fetch::<f32>(value_ft).expect("Can't fetch output");
    // Copy data over.
    policy.copy_from_slice(&policy_output[0..64 * 64]);
    *value = value_output[0];
    println!("value: {}", value);
  }
}

use engine::nnue::Nnue;
use engine::rules::{State, Move};

fn main() {
  let mut state = State::starting_state();
  let mut nnue = Nnue::new(&state, engine::nnue::BUNDLED_NETWORK);
  let moves = [
    "e2e4", "a3a3",
    "e7e5", "a3a4",
    //"d1h5", "a4a3",
    //"g7g6", "a3a4",
    //"e1e2", "a4a3",
    //"g6h5",
    //"f1a6", "a4a3",
    //"b7a6",
  ];
  for uci in moves {
    let m = Move::from_uci(uci).unwrap();
    // Print the move in json.
    println!("{}", serde_json::to_string(&m).unwrap());
    state.apply_move::<true>(m, Some(&mut nnue)).unwrap();
  }
  println!("Initial hash: {:016x}", nnue.get_debugging_hash());
  nnue.dump_state();
  let eval = nnue.evaluate(&state);
  println!("Eval: {}", eval);
}

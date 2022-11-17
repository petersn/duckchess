use engine::nnue::Nnue;
use engine::rules::State;

// Implement perft recursively
fn perft(nnue: &mut Nnue, state: &State, depth: usize) -> u64 {
  if depth == 0 {
    return 1;
  }
  let mut count = 0;
  let mut moves = vec![];
  state.move_gen::<false>(&mut moves);
  for m in moves {
    //let nnue_hash = nnue.get_debugging_hash();
    let mut child = state.clone();
    let undo_cookie = child.apply_move::<true>(m, Some(nnue)).unwrap();
    let _eval = nnue.evaluate(&child);
    count += perft(nnue, &child, depth - 1);
    nnue.undo(undo_cookie);
    //assert_eq!(nnue_hash, nnue.get_debugging_hash());
  }
  count
}

fn main() {
  let start_time = std::time::Instant::now();
  let state = State::starting_state();
  let mut nnue = Nnue::new(&state, engine::nnue::BUNDLED_NETWORK);
  let nodes = perft(&mut nnue, &state, 5);
  let elapsed = start_time.elapsed().as_secs_f32();
  println!("{} nodes", nodes);
  println!("{} seconds", elapsed);
  println!("{} Mnodes/second", 1e-6 * nodes as f32 / elapsed);
}

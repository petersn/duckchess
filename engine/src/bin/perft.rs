use engine::nnue::Nnue;
use engine::rules::State;

// Implement perft recursively
fn perft(nnue: &mut Nnue, state: &State, depth: usize) -> u64 {
  if depth == 0 {
    return 1;
  }
  //let indent = " ".repeat(2 * (depth - 1));
  let mut count = 0;
  let mut moves = vec![];
  state.move_gen::<false>(&mut moves);
  // Sample just a few moves.
  use rand::seq::SliceRandom;
  let mut rng = rand::thread_rng();
  moves.shuffle(&mut rng);
  moves.truncate(2);
  for m in moves {
    let starting_nnue_hash = nnue.get_debugging_hash();
    let mut child = state.clone();
    let adjustment = child.apply_move::<true>(m, Some(nnue)).unwrap();
    //println!("{}{} {:?}", indent, m, adjustment);
    //let _eval = nnue.evaluate(&child);
    count += perft(nnue, &child, depth - 1);
    nnue.apply_adjustment::<true>(state, &adjustment);
    let after_undo_hash = nnue.get_debugging_hash();
    // Recompute.
    nnue.recompute_linear_state(state);
    let recompute_hash = nnue.get_debugging_hash();
    //println!("{}{} {} {}", indent, starting_nnue_hash, after_undo_hash, recompute_hash);
    //println!("{}Undoing {}", indent, m);
    assert_eq!(starting_nnue_hash, after_undo_hash);
    assert_eq!(starting_nnue_hash, recompute_hash);
  }
  count
}

fn main() {
  let start_time = std::time::Instant::now();
  let state = State::starting_state();
  let mut nnue = Nnue::new(&state, engine::nnue::BUNDLED_NETWORK);
  let nodes = perft(&mut nnue, &state, 25);
  let elapsed = start_time.elapsed().as_secs_f32();
  println!("{} nodes", nodes);
  println!("{} seconds", elapsed);
  println!("{} Mnodes/second", 1e-6 * nodes as f32 / elapsed);
}

use engine::nnue::{UndoCookie, Nnue};
use engine::rules::{Move, State};

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
    let eval = nnue.evaluate(&child);
    count += perft(nnue, &child, depth - 1);
    nnue.undo(undo_cookie);
    //assert_eq!(nnue_hash, nnue.get_debugging_hash());
  }
  count
}

// struct StackEntry {
//   depth: usize,
//   state: State,
//   m:     Move,
// }

fn main() {
  let start_time = std::time::Instant::now();
  let state = State::starting_state();
  let mut nnue = Nnue::new(&state);
  let nodes = perft(&mut nnue, &state, 4);
  // let mut nodes = 0;
  // let mut moves = vec![];
  // let mut stack = vec![StackEntry {
  //   depth: 0,
  //   state: State::starting_state(),
  //   m:     Move::from_uci("e2e4").unwrap(),
  // }];
  // let mut nnue = engine::nnue::Nnue::new(&State::starting_state());
  // while let Some(mut entry) = stack.pop() {
  //   nodes += 1;
  //   let starting_zobrist = entry.state.zobrist;
  //   let nnue_hash = nnue.get_debugging_hash();
  //   let undo_cookie = entry.state.apply_move::<true>(entry.m, Some(&mut nnue)).unwrap();
  //   //nnue.evaluate(&entry.state);
  //   if entry.depth != 5 {
  //     entry.state.move_gen::<false>(&mut moves);
  //     for (i, m) in moves.iter().enumerate() {
  //       //let mut new_state = entry.state.clone();
  //       //let undo_cookie = new_state.apply_move::<false>(*m, None).unwrap();
  //       stack.push(StackEntry {
  //         depth: entry.depth + 1,
  //         state: entry.state.clone(),
  //         m:     *m,
  //       });
  //     }
  //     moves.clear();
  //   }
  //   nnue.undo(undo_cookie);
  //   entry.state.undo(undo_cookie);
  //   assert_eq!(entry.state.zobrist, starting_zobrist);
  //   assert_eq!(nnue_hash, nnue.get_debugging_hash());
  // }
  let elapsed = start_time.elapsed().as_secs_f32();
  println!("{} nodes", nodes);
  println!("{} seconds", elapsed);
  println!(
    "{} Mnodes/second",
    1e-6 * nodes as f32 / elapsed
  );
}

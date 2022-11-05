use engine::rules::{State, Move};
use engine::nnue::UndoCookie;

struct StackEntry {
  depth: usize,
  state: State,
  m: Move,
}

fn main() {
  let start_time = std::time::Instant::now();
  let mut nodes = 0;
  let mut moves = vec![];
  let mut stack = vec![
    StackEntry {
      depth: 0,
      state: State::starting_state(),
      m: Move::from_uci("e2e4").unwrap(),
    }
  ];
  let mut nnue = engine::nnue::Nnue::new(&State::starting_state());
  while let Some(mut entry) = stack.pop() {
    nodes += 1;
    let undo_cookie = entry.state.apply_move::<true>(entry.m, Some(&mut nnue)).unwrap();
    nnue.evaluate(&entry.state);
    if entry.depth == 4 {
      
    } else {
      entry.state.move_gen::<false>(&mut moves);
      for (i, m) in moves.iter().enumerate() {
        //let mut new_state = entry.state.clone();
        //let undo_cookie = new_state.apply_move::<false>(*m, None).unwrap();
        stack.push(StackEntry {
          depth: entry.depth + 1,
          state: entry.state.clone(),
          m: *m,
        });
      }
      moves.clear();
    }
    nnue.undo(undo_cookie);
  }
  println!("{} nodes", nodes);
  println!("{} seconds", start_time.elapsed().as_secs_f64());
  println!("{} Mnodes/second", 1e-6 * nodes as f64 / start_time.elapsed().as_secs_f64());
}

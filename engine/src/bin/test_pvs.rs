use std::io::BufRead;

use engine::rules::{Move, Player};

fn main() {
  let mut e = engine::search::Engine::new(1234, 64 * 1024 * 1024, false);
  let start_time = std::time::Instant::now();
  let mut state = e.get_state_mut();

  // We now load up a game and run to a fixed depth from each position.
  let f = std::fs::File::open("run-012/eval-games/games-mcts-3f5361237eb22fc6.json").unwrap();
  // Read one line.
  let mut reader = std::io::BufReader::new(f);
  let mut line = String::new();
  reader.read_line(&mut line).unwrap();
  // Parse the line as JSON.
  let value = serde_json::from_str::<serde_json::Value>(&line).unwrap();
  // Print out all of the moves in value["moves"].
  let moves = value["moves"].as_array().unwrap();
  for (i, m) in moves.iter().enumerate() {
    let from = m["from"].as_u64().unwrap() as u8;
    let to = m["to"].as_u64().unwrap() as u8;
    let move_ = Move { from, to };
    //let pv = e.run(4, true);
    let pv = e.run(5, true);
    println!("Move {}: {:?}", i, move_);
    e.apply_move(move_).unwrap();
  }

  //state.pawns[Player::Black as usize].0 = 0x0000000000000000;
  //state.knights[Player::Black as usize].0 = 0x0000000000000000;
  //state.bishops[Player::Black as usize].0 = 0x0000000000000000;
  //state.rooks[Player::Black as usize].0 = 0x0000000000000000;
  //state.queens[Player::Black as usize].0 = 0x0000000000000000;
  ////state.queens[Player::White as usize].0 = 0x0000000000550000;
  //state.pawns[Player::White as usize].0 = 0x0000000000000000;
  //state.queens[Player::White as usize].0 = 0x0055000000000000;

  // We now evaluate in a battery of test positions.

  //e.apply_move(Move { from: 1, to: 16 });
  //let (score, move_pair) = e.mate_search(9);
  let elapsed = start_time.elapsed().as_secs_f32();
  //println!("Score: {}", pv.eval);
  //println!("Moves: {:?}", pv.moves);
  println!("Elapsed: {:.3}s", elapsed);
  println!("Nodes: {} M", e.nodes_searched as f32 / 1e6);
  println!("Nodes/s: {} M", e.nodes_searched as f32 / elapsed / 1e6);
}

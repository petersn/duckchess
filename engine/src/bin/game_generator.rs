use std::io::Write;

fn work() {
  // Open a file for writing
  let output_path = format!("games/games-{:016x}.json", rand::random::<u64>());
  let mut output_file = std::fs::File::create(output_path).unwrap();
  loop {
    let mut moves = vec![];
    let mut engine = engine::new_engine(rand::random());
    for _ in 0..300 {
      let p = engine::run_internal(&mut engine, 4);
      if let Some(m) = p.1.0 {
        moves.push(m);
        engine::apply_move_internal(&mut engine, m);
      } else {
        break;
      }
    }
    let outcome = engine::get_outcome_internal(&engine);
    println!("Game generated: moves={} outcome={:?}", moves.len(), outcome);
    let obj = serde_json::json!({
      "outcome": outcome,
      "moves": moves,
    });
    let s = serde_json::to_string(&obj).unwrap();
    output_file.write_all(s.as_bytes()).unwrap();
    output_file.write_all(b"\n").unwrap();
    output_file.flush().unwrap();
  }
}

fn main() {
  // Launch workers for every thread.
  let mut workers = vec![];
  for _ in 0..num_cpus::get() {
    workers.push(std::thread::spawn(work));
  }
  // Wait for all workers to finish.
  for w in workers {
    w.join().unwrap();
  }
}

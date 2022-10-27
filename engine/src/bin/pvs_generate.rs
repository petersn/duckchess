use std::io::Write;

use clap::Parser;
use rand::seq::SliceRandom;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  output_dir: String,
}

fn work(output_dir: &str) {
  // Open a file for writing
  let output_path = format!("{}/games-{:016x}.json", output_dir, rand::random::<u64>());
  let mut output_file = std::fs::File::create(output_path).unwrap();
  let mut rng = rand::thread_rng();
  loop {
    let mut moves = vec![];
    let mut was_rand = vec![];
    let mut engine = engine::search::Engine::new(rand::random());
    for move_index in 0..300 {
      let r: f32 = rand::random();
      let random_move = move_index == 0 || if move_index < 6 { r < 0.5 } else { r < 0.01 };
      let p = match random_move {
        true => engine.get_moves().choose(&mut rng).map(|x| *x),
        false => engine.run(4).1 .0,
      };
      if let Some(m) = p {
        was_rand.push(random_move);
        moves.push(m);
        engine.apply_move(m).unwrap();
      } else {
        break;
      }
    }
    let outcome = engine.get_outcome().to_option_str();
    println!(
      "Game generated: moves={} outcome={:?}",
      moves.len(),
      outcome,
    );
    let obj = serde_json::json!({
      "outcome": outcome,
      "moves": moves,
      "was_rand": was_rand,
      "version": 4,
    });
    let s = serde_json::to_string(&obj).unwrap();
    output_file.write_all(s.as_bytes()).unwrap();
    output_file.write_all(b"\n").unwrap();
    output_file.flush().unwrap();
  }
}

fn main() {
  let args = Args::parse();
  let output_dir: &'static str = Box::leak(String::into_boxed_str(args.output_dir));
  println!("Writing to {}", output_dir);

  // Launch workers for every thread.
  let mut workers = vec![];
  for _ in 0..(num_cpus::get() - 3) {
    workers.push(std::thread::spawn(move || work(output_dir)));
  }
  // Wait for all workers to finish.
  for w in workers {
    w.join().unwrap();
  }
}

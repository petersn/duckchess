use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

use clap::Parser;

const PLAYOUT_CAP_RANDOMIZATION_P: f32 = 0.25;
const LARGE_PLAYOUTS: u32 = 1600;
const SMALL_PLAYOUTS: u32 = 200;
const GAME_LEN_LIMIT: usize = 300;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  output_dir: String,

  #[arg(short, long)]
  model_dir: String,
}

#[tokio::main]
async fn main() {
  // Parse command line arguments.
  let args = Args::parse();

  let inference_engine: &InferenceEngine = Box::leak(Box::new(InferenceEngine::create(&args.model_dir).await));

  let output_path = format!("{}/games-{:016x}.json", args.output_dir, rand::random::<u64>());
  println!("Writing to {}", output_path);
  let output_file: &'static _ = Box::leak(Box::new(tokio::sync::Mutex::new(
    std::fs::File::create(output_path).unwrap(),
  )));

  // Spawn several tasks to run MCTS in parallel.
  let mut tasks = Vec::new();
  for task_id in 0..2 * engine::inference::BATCH_SIZE - 1 {
    tasks.push(tokio::spawn(async move {
      use std::io::Write;
      loop {
        let mut mcts = Mcts::create(inference_engine).await;
        let mut moves = vec![];
        let mut train_moves = vec![];
        let mut was_large = vec![];
        let mut steps_performed = vec![];
        for _ in 0..GAME_LEN_LIMIT {
          let full_search = rand::random::<f32>() < PLAYOUT_CAP_RANDOMIZATION_P;
          was_large.push(full_search);
          let playouts = match full_search {
            true => LARGE_PLAYOUTS,
            false => SMALL_PLAYOUTS,
          };
          // KataGo paper only adds noise on full searches
          if full_search {
            mcts.apply_noise_to_root();
          }
          let steps = mcts.step_until(playouts).await;
          //println!("{}: tree={} steps={}", task_id, playouts, steps);
          steps_performed.push(steps);
          let train_move = match full_search {
            true => mcts.select_train_move(),
            false => None,
          };
          train_moves.push(train_move);
          match mcts.sample_move_by_visit_count() {
            None => break,
            Some(m) => {
              //println!("[{task_id}] Generated a move: {:?}", m);
              moves.push(m);
              mcts.apply_move(m).await;
            }
          }
        }
        let total_steps = steps_performed.iter().sum::<u32>();
        println!("[{task_id}] Generated a game len={} total-steps={} steps-per-move={:.2}", moves.len(), total_steps, total_steps as f32 / moves.len() as f32);
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mcts.get_state().get_outcome().to_str(),
            "moves": moves,
            "train_moves": train_moves,
            "was_large": was_large,
            "steps_performed": steps_performed,
            "playout_cap_randomization_p": PLAYOUT_CAP_RANDOMIZATION_P,
            "large_playouts": LARGE_PLAYOUTS,
            "small_playouts": SMALL_PLAYOUTS,
            "game_len_limit": GAME_LEN_LIMIT,
            "version": 101,
          });
          let s = serde_json::to_string(&obj).unwrap();
          file.write_all(s.as_bytes()).unwrap();
          file.write_all(b"\n").unwrap();
          file.flush().unwrap();
        }
      }
      //println!("Starting task {} (on thread: {:?})", task_id, std::thread::current().id());
      //println!("[{task_id}] Completed init");
      //for step in 0..1_000_000 {
      //  //println!("Task {} step {} (on thread: {:?})", task_id, step, std::thread::current().id());
      //  mcts.step().await;
      //}
      //println!("[{task_id}] Completed steps");
    }));
  }
  // Join all the tasks.
  for task in tasks {
    task.await.unwrap();
  }
  //println!("Time elapsed in expensive_function() is: {:?}", elapsed);
  //mcts.print_tree();
}

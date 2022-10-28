use clap::Parser;
use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

const GAME_LEN_LIMIT: usize = 300;

trait DuckChessEngine {
  fn create() -> Self;
  fn think(&mut self);
  fn get_move(&self) -> Option<engine::rules::Move>;
  fn apply_move(&mut self, m: engine::rules::Move);
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  playouts: u32,

  #[arg(long)]
  model1_dir: String,

  #[arg(long)]
  model2_dir: String,

  #[arg(long)]
  opening_randomization: bool,

  #[arg(short, long)]
  output_dir: String,
}

#[tokio::main]
async fn main() {
  let args = Args::parse();

  let model1_dir: &'static str = Box::leak(String::into_boxed_str(args.model1_dir));
  let model2_dir: &'static str = Box::leak(String::into_boxed_str(args.model2_dir));

  let inference_engine1: &InferenceEngine =
    Box::leak(Box::new(InferenceEngine::create(model1_dir).await));
  let inference_engine2: Option<&InferenceEngine> = match model2_dir == "@pvs" {
    true => None,
    false => Some(Box::leak(Box::new(InferenceEngine::create(model2_dir).await))),
  };

  let output_path = format!(
    "{}/games-mcts-{:016x}.json",
    args.output_dir,
    rand::random::<u64>()
  );
  println!("Writing to {}", output_path);
  let output_file: &'static _ = Box::leak(Box::new(tokio::sync::Mutex::new(
    std::fs::File::create(output_path).unwrap(),
  )));

  // Spawn several tasks to run MCTS in parallel.
  let mut tasks = Vec::new();
  for task_id in 0..2 * engine::inference::BATCH_SIZE - 1 {
    tasks.push(tokio::spawn(async move {
      use std::io::Write;
      let mut engine1_is_white = rand::random::<bool>();
      loop {
        // Alternate who starts as white.
        engine1_is_white = !engine1_is_white;
        let mcts1 = Mcts::create(inference_engine1).await;
        let mcts2 = Mcts::create(inference_engine2.unwrap()).await;
        let (white_engine_name, black_engine_name) = match engine1_is_white {
          true => (model1_dir, model2_dir),
          false => (model2_dir, model1_dir),
        };
        let mut mctses = match engine1_is_white {
          true => [mcts1, mcts2],
          false => [mcts2, mcts1],
        };

        // This array tracks the moves that actually occur in the game.
        let mut moves: Vec<engine::rules::Move> = vec![];
        // This array tracks how many steps were actually performed to compute this move.
        let mut steps_performed: Vec<u32> = vec![];

        for move_number in 0..GAME_LEN_LIMIT {
          let index = (move_number / 2) % 2;
          let mcts = &mut mctses[index];
          let steps = mcts.step_until(args.playouts, true).await;
          let game_move = mcts.sample_move_by_visit_count();
          match game_move {
            None => break,
            Some(game_move) => {
              steps_performed.push(steps);
              moves.push(game_move);
              mctses[0].apply_move(game_move).await;
              mctses[1].apply_move(game_move).await;
            }
          }
        }
        let total_steps = steps_performed.iter().sum::<u32>();
        println!(
          "[{task_id}] Generated a game len={} total-steps={} steps-per-move={:.2}",
          moves.len(),
          total_steps,
          total_steps as f32 / moves.len() as f32,
        );
        // Guarantee that all lists are the same length.
        assert_eq!(moves.len(), steps_performed.len());
        // Make sure the engines agree about who won/
        assert_eq!(
          mctses[0].get_state().get_outcome(),
          mctses[1].get_state().get_outcome(),
        );
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mctses[0].get_state().get_outcome().to_option_str(),
            "final_state": mctses[0].get_state(),
            "moves": moves,
            "steps_performed": steps_performed,
            "engine_white": white_engine_name,
            "engine_black": black_engine_name,
            "playouts_white": args.playouts,
            "playouts_black": args.playouts,
            "game_len_limit": GAME_LEN_LIMIT,
            "version": "mcts-compete-1",
          });
          let s = serde_json::to_string(&obj).unwrap();
          file.write_all(s.as_bytes()).unwrap();
          file.write_all(b"\n").unwrap();
          file.flush().unwrap();
        }
      }
    }));
  }
  // Join all the tasks.
  for task in tasks {
    task.await.unwrap();
  }
}

use clap::Parser;
use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

const PLAYOUT_CAP_RANDOMIZATION_P: f32 = 0.25;
const FULL_SEARCH_PLAYOUTS: u32 = 1000;
const SMALL_SEARCH_PLAYOUTS: u32 = 200;
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
  let args = Args::parse();

  let inference_engine: &InferenceEngine =
    Box::leak(Box::new(InferenceEngine::create(&args.model_dir).await));

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
      loop {
        let mut mcts = Mcts::create(inference_engine).await;
        // This array tracks the moves that actually occur in the game.
        let mut moves: Vec<engine::rules::Move> = vec![];
        // This array contains the training targets for the policy head.
        let mut train_dists: Vec<Option<Vec<(engine::rules::Move, f32)>>> = vec![];
        // This array says if we attempted a full search to choose a training move.
        let mut full_search: Vec<bool> = vec![];
        // This array tracks how many steps were actually performed to compute this move.
        let mut steps_performed: Vec<u32> = vec![];

        for _ in 0..GAME_LEN_LIMIT {
          // Decide whether or not to perform a full search.
          let do_full_search = rand::random::<f32>() < PLAYOUT_CAP_RANDOMIZATION_P;
          let playouts = match do_full_search {
            true => FULL_SEARCH_PLAYOUTS,
            false => SMALL_SEARCH_PLAYOUTS,
          };
          // KataGo paper only adds noise on full searches.
          if do_full_search {
            mcts.apply_noise_to_root();
          }
          // Perform the actual tree search.
          // We early out only if we're not doing a full search, to properly compute our training target.
          let steps = mcts.step_until(playouts, !do_full_search).await;
          //println!("{}: tree={} steps={}", task_id, playouts, steps);

          match mcts.sample_move_by_visit_count() {
            None => break,
            Some(game_move) => {
              //println!("[{task_id}] Generated a move: {:?}", m);
              full_search.push(do_full_search);
              train_dists.push(match do_full_search {
                true => Some(mcts.get_train_distribution()),
                false => None,
              });
              steps_performed.push(steps);
              moves.push(game_move);
              mcts.apply_move(game_move).await;
            }
          }
        }
        let total_steps = steps_performed.iter().sum::<u32>();
        println!(
          "[{task_id}] Generated a game len={} total-steps={} steps-per-move={:.2}",
          moves.len(),
          total_steps,
          total_steps as f32 / moves.len() as f32
        );
        // Guarantee that all lists are the same length.
        assert_eq!(moves.len(), train_dists.len());
        assert_eq!(moves.len(), full_search.len());
        assert_eq!(moves.len(), steps_performed.len());
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mcts.get_state().get_outcome().to_option_str(),
            "final_state": mcts.get_state(),
            "moves": moves,
            "train_dists": train_dists,
            "full_search": full_search,
            "steps_performed": steps_performed,
            "playout_cap_randomization_p": PLAYOUT_CAP_RANDOMIZATION_P,
            "full_search_playouts": FULL_SEARCH_PLAYOUTS,
            "small_search_playouts": SMALL_SEARCH_PLAYOUTS,
            "game_len_limit": GAME_LEN_LIMIT,
            "version": "mcts-1",
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

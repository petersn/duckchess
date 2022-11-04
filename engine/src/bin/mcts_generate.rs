use clap::Parser;
use engine::inference::InferenceEngine;
use engine::inference::ModelOutputs;
use engine::inference_desktop::TensorFlowEngine;
use engine::mcts::Mcts;
use engine::mcts::PendingPath;
use tokio::sync::mpsc;
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::Mutex;

const PLAYOUT_CAP_RANDOMIZATION_P: f32 = 0.25;
//const FULL_SEARCH_PLAYOUTS: u32 = 1000;
//const SMALL_SEARCH_PLAYOUTS: u32 = 200;
const FULL_SEARCH_PLAYOUTS: u32 = 600;
const SMALL_SEARCH_PLAYOUTS: u32 = 100;
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

  let inference_engine: &TensorFlowEngine<(usize, PendingPath)> =
    Box::leak(Box::new(TensorFlowEngine::new(&args.model_dir)));

  let output_path = format!(
    "{}/games-mcts-{:016x}.json",
    args.output_dir,
    rand::random::<u64>()
  );
  println!("Writing to {}", output_path);
  let output_file: &'static _ = Box::leak(Box::new(Mutex::new(
    std::fs::File::create(output_path).unwrap(),
  )));

  let (tx_channels, rx_channels): (Vec<_>, Vec<_>) = (0..4
    * TensorFlowEngine::<()>::DESIRED_BATCH_SIZE)
    .map(|_| tokio::sync::mpsc::unbounded_channel())
    .unzip();
  let tx_channels: &'static _ = Box::leak(Box::new(tx_channels));

  let mut tasks = Vec::new();

  // Spawn some tasks to do inference.
  for _ in 0..2 {
    tasks.push(tokio::spawn(async move {
      let mut last_print = std::time::Instant::now();
      let mut evals_since_last_print = 0;
      loop {
        inference_engine.semaphore.acquire().await.unwrap().forget();
        // Check if we should be doing work here.
        inference_engine.predict(|inference_results| {
          evals_since_last_print += inference_results.cookies.len();
          for (i, (cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
            //let node_id = inference_engine.node_id(i);
            //let node = mcts.node_mut(node_id);
            //node.inference_result = Some(inference_result.clone());
            //notifiers[node_id].notify_waiters();
            let tx = &tx_channels[*cookie];
            // FIXME: It should be possible to remove this clone.
            tx.send((pending_path.clone(), inference_results.get(i)))
              .map_err(|_| "error sending")
              .unwrap();
          }
        });
        if last_print.elapsed().as_secs() >= 20 {
          println!(
            "Evals per second: {}",
            evals_since_last_print as f32 / last_print.elapsed().as_secs_f32()
          );
          evals_since_last_print = 0;
          last_print = std::time::Instant::now();
        }
      }
    }));
  }

  // Spawn several tasks to run MCTS in parallel.
  for (task_id, mut rx) in rx_channels.into_iter().enumerate() {
    tasks.push(tokio::spawn(async move {
      use std::io::Write;
      loop {
        let seed = rand::random::<u64>();
        let mut mcts = Mcts::new(task_id, seed, inference_engine);
        // This array tracks the moves that actually occur in the game.
        let mut moves: Vec<engine::rules::Move> = vec![];
        // This array says if the game move was uniformly random, or based on our tree search.
        let mut was_rand: Vec<bool> = vec![];
        // This array contains the training targets for the policy head.
        let mut train_dists: Vec<Option<Vec<(engine::rules::Move, f32)>>> = vec![];
        // This array says if we attempted a full search to choose a training move.
        let mut full_search: Vec<bool> = vec![];
        // This array tracks how many steps were actually performed to compute this move.
        let mut steps_performed: Vec<u32> = vec![];

        for move_number in 0..GAME_LEN_LIMIT {
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
          let mut steps = 0;
          for _ in 0..playouts {
            if !do_full_search && mcts.have_reached_visit_count(playouts) {
              break;
            }
            steps += 1;
            mcts.step();
            // Make sure we fully flush here.
            while mcts.any_in_flight() {
              let (pending_path, model_outputs) = rx.recv().await.unwrap();
              mcts.process_path(pending_path, model_outputs);
            }
          }
          //let steps = mcts.step_until(playouts, !do_full_search).await;
          //println!("{}: tree={} steps={}", task_id, playouts, steps);

          // We pick a uniformly random move for 10% of opening moves
          // for each player, then 5% of second moves for each player.
          let uniformly_random_move_prob = 0.10 - (0.05 * (move_number / 4) as f32);
          let pick_randomly = rand::random::<f32>() < uniformly_random_move_prob;
          let game_move = match pick_randomly {
            true => {
              use rand::seq::SliceRandom;
              let mut moves = vec![];
              mcts.get_state().move_gen::<false>(&mut moves);
              moves.choose(&mut rand::thread_rng()).map(|m| *m)
            }
            false => mcts.sample_move_by_visit_count(),
          };

          match game_move {
            None => break,
            Some(game_move) => {
              //println!("[{task_id}] Generated a move: {:?}", m);
              was_rand.push(pick_randomly);
              full_search.push(do_full_search);
              train_dists.push(match do_full_search {
                true => Some(mcts.get_train_distribution()),
                false => None,
              });
              steps_performed.push(steps);
              moves.push(game_move);
              mcts.apply_move(game_move);
              mcts.get_state().sanity_check().unwrap();
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
        assert_eq!(moves.len(), was_rand.len());
        assert_eq!(moves.len(), train_dists.len());
        assert_eq!(moves.len(), full_search.len());
        assert_eq!(moves.len(), steps_performed.len());
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mcts.get_state().get_outcome().map(|o| o.to_str()),
            "final_state": mcts.get_state(),
            "moves": moves,
            "was_rand": was_rand,
            "train_dists": train_dists,
            "full_search": full_search,
            "steps_performed": steps_performed,
            "playout_cap_randomization_p": PLAYOUT_CAP_RANDOMIZATION_P,
            "full_search_playouts": FULL_SEARCH_PLAYOUTS,
            "small_search_playouts": SMALL_SEARCH_PLAYOUTS,
            "game_len_limit": GAME_LEN_LIMIT,
            "seed": seed,
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

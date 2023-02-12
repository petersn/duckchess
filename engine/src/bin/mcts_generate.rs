use clap::Parser;
use engine::inference::InferenceEngine;
use engine::inference_tensorrt::TensorRTEngine;
use engine::mcts::Mcts;
use engine::mcts::PendingPath;
use engine::mcts::SearchParams;
use engine::rules::State;
use tokio::io::AsyncBufReadExt;
use tokio::sync::Mutex;

const PLAYOUT_CAP_RANDOMIZATION_P: f32 = 0.25;
const FULL_SEARCH_PLAYOUTS: u32 = 1500;
const SMALL_SEARCH_PLAYOUTS: u32 = 250;
//const FULL_SEARCH_PLAYOUTS: u32 = 600;
//const SMALL_SEARCH_PLAYOUTS: u32 = 100;
const GAME_LEN_LIMIT: usize = 800;

const ALPHA_BETA_SEARCH_DEPTH: u16 = 2;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  output_dir: Option<String>,

  #[arg(short, long)]
  model_dir: Option<String>,

  #[arg(short, long, default_value = "default")]
  search_params: SearchParams,

  #[arg(short, long)]
  batch_size: usize,

  #[arg(short, long)]
  signals_dir: Option<String>,
}

fn check_signals_dir(signals_dir: &str, previously_processed: &mut String) -> Option<(String, String)> {
  // Enumerate all files in the signals directory.
  let files = std::fs::read_dir(signals_dir).unwrap();
  // Take the one with the lexicographically last path, that starts with "signal-".
  let most_recent = files
    .filter_map(|f| {
      let f = f.unwrap();
      let path = f.path().to_str().unwrap().to_string();
      if path.starts_with("signal-") {
        Some(path)
      } else {
        None
      }
    })
    .max();
  if let Some(most_recent) = most_recent {
    // Check if this is more recent than our most recent.
    if most_recent > *previously_processed {
      println!("\x1b[96mFound new signal file:\x1b[0m {}", most_recent);
      // If so, read the contents of the file.
      let contents = std::fs::read_to_string(&most_recent).unwrap();
      assert!(contents.starts_with("skip"));
      // Split the line at the two :::s.
      let mut parts = contents.split(":::");
      parts.next();
      let new_model_path = parts.next().unwrap();
      let new_output_dir = parts.next().unwrap();
      assert!(parts.next().is_none());
      *previously_processed = most_recent;
      return Some((new_model_path.to_string(), new_output_dir.to_string()));
    }
  }
  None
}

#[tokio::main]
async fn main() {
  let args = Args::parse();

  // We figure out the initial model, either from the command line or from the signals directory.
  let mut previously_processed = String::new();
  let (model_dir, output_dir) = args.signals_dir.as_ref()
    .and_then(|signals_dir| check_signals_dir(signals_dir, &mut previously_processed))
    .unwrap_or((args.model_dir.clone().unwrap(), args.output_dir.clone().unwrap()));

  println!("Using model dir: {}", model_dir);
  println!("Using output dir: {}", output_dir);

  //let batch_size = TensorRTEngine::<()>::DESIRED_BATCH_SIZE;
  let batch_size = args.batch_size;
  println!("Using batch size: {}", batch_size);

  let inference_engine: &TensorRTEngine<(usize, PendingPath)> =
    Box::leak(Box::new(TensorRTEngine::new(batch_size, &model_dir)));

  let search_params: &'static SearchParams = Box::leak(Box::new(args.search_params));

  println!("Search params: {:?}", search_params);
  println!("Alpha-beta depth: {}", ALPHA_BETA_SEARCH_DEPTH);

  let create_output_file = |output_dir: &str| {
    let output_path = format!(
      "{}/games-mcts-{:016x}.json",
      output_dir,
      rand::random::<u64>()
    );
    println!("Writing to {}", output_path);
    std::fs::File::create(output_path).unwrap()
  };

  let output_file: &'static _ =
    Box::leak(Box::new(Mutex::new(create_output_file(&output_dir))));

  let (tx_channels, rx_channels): (Vec<_>, Vec<_>) =
    (0..5 * batch_size).map(|_| tokio::sync::mpsc::unbounded_channel()).unzip();
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
        let model_name_start = inference_engine.get_current_model_name();
        let seed = rand::random::<u64>();
        let mut mcts = Mcts::new(
          task_id,
          seed,
          inference_engine,
          search_params.clone(),
          State::starting_state(),
          false,
        );
        // This array tracks the moves that actually occur in the game.
        let mut moves: Vec<engine::rules::Move> = vec![];
        // This array gives a few random example serialized states.
        let mut states = vec![];
        // This array says if the game move was uniformly random, or based on our tree search.
        let mut was_rand: Vec<bool> = vec![];
        // This array says if our little alpha-beta search resulted in a move that was used.
        let mut used_mate_move: Vec<bool> = vec![];
        // This array says what the alpha-beta search evaluation was.
        let mut mate_search_evals: Vec<i32> = vec![];
        // This array tracks pairs (bad_move, non_bad_move) of whether in move sampling we considered any
        // moves that hang mate in 1, and if we considered any moves that don't hang mate in 1.
        let mut bad_non_bad_moves: Vec<(bool, bool)> = vec![];
        // Array of depths to which we ran the exhaustive alpha-beta search for mates, by position.
        let mut alpha_beta_search_depths: Vec<u16> = vec![];
        // This array contains the training targets for the policy head.
        let mut train_dists: Vec<Option<Vec<(engine::rules::Move, f32)>>> = vec![];
        // We also store all of the visit distributions, just for debugging for now.
        let mut visit_dists: Vec<Vec<(engine::rules::Move, i32)>> = vec![];
        // This array contains the average value of the entire MCTS tree (rooted at the current state) at each move.
        let mut root_values: Vec<f32> = vec![];
        // This array contains the size of the entire MCTS tree (rooted at the current state) at each move.
        let mut root_visits: Vec<u32> = vec![];
        // This array says if we attempted a full search to choose a training move.
        let mut full_search: Vec<bool> = vec![];
        // This array tracks how many steps were actually performed to compute this move.
        let mut steps_performed: Vec<u32> = vec![];

        let mut just_kings_count = 0;
        let mut was_adjudicated_early = false;

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

          // Start by performing a small exact tree search for mates.
          let alpha_beta_depth = match mcts.get_state().is_duck_move {
            true => 2 * ALPHA_BETA_SEARCH_DEPTH,
            false => 2 * ALPHA_BETA_SEARCH_DEPTH + 1,
          };
          let (alpha_beta_eval, alpha_beta_moves) = if mcts.get_state().get_outcome().is_some() {
            (0, None)
          } else {
            engine::search::mate_search(
              &mcts.get_state(),
              alpha_beta_depth,
            )
          };
          if alpha_beta_eval > 0 && alpha_beta_moves.is_none() {
            eprintln!("\x1b[91mWEIRD BUG!!!!!!!!!!!!!!!!!!!!!!! State: {:?}\x1b[0m", mcts.get_state());
          }
          // If we have a winning move then just play it.
          let mating_move = if alpha_beta_eval > 0 && alpha_beta_moves.is_some() {
            let mating_move = alpha_beta_moves.unwrap().0;
            // Assert that it's legal.
            assert!(mcts.check_if_move_is_legal_from_root(mating_move));
            Some(mating_move)
          } else {
            None
          };

          // Perform the actual tree search.
          // We early out only if we're not doing a full search, to properly compute our training target.
          let mut steps = 0;
          for _ in 0..playouts {
            // For training moves we need the full 1000 visits to have an accurate distribution.
            // But for fast moves we can stop once we know that the second most visited move can't surpass the most.
            if if do_full_search {
              mcts.have_reached_visit_count(playouts)
            } else {
              mcts.have_reached_visit_count_short_circuiting(playouts)
            } {
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
          // let uniformly_random_move_prob = 0.10 - (0.05 * (move_number / 4) as f32);
          // let (full_search_beta, fast_search_beta) = (2.0, 4.0);

          // // FIXME: I disabled the new randomization and temperature stuff for now.
          // Adjustment: I found that random moves in the opening were hanging mates, which seems a little much.
          // So I'll just not randomize via uniform randomness after the first four plies.
          // However, in place of that I'll lower the temperatures for the first sixteen plies.
          let uniformly_random_move_prob = match move_number < 4 {
            true => 0.05,
            false => 0.0,
          };
          let (full_search_beta, fast_search_beta) = if move_number < 8 {
            (0.75, 1.0)
          } else if move_number < 16 {
            (1.0, 1.5)
          } else if move_number < 32 {
            (2.0, 4.0)
          } else {
            (4.0, 4.0)
          };

          let pick_randomly = rand::random::<f32>() < uniformly_random_move_prob;
          let (game_move, bad_move, non_bad_move) = match (mating_move, pick_randomly) {
            (Some(m), _) => (Some(m), false, true),
            (None, true) => {
              use rand::seq::SliceRandom;
              let mut moves = vec![];
              mcts.get_state().move_gen::<false>(&mut moves);
              (moves.choose(&mut rand::thread_rng()).map(|m| *m), false, false)
            }
            // We use temperature = 0.5 for training moves, and temperature = 0.25 for fast moves, to improve play strength.
            (None, false) => match do_full_search {
              true => mcts.sample_move_by_visit_count(full_search_beta, true),
              false => mcts.sample_move_by_visit_count(fast_search_beta, true),
            },
          };

          match game_move {
            None => break,
            Some(game_move) => {
              states.push(match rand::random::<f32>() < 0.01 {
                true => Some(mcts.get_state().clone()),
                false => None,
              });
              // Make sure the move is actually in the list of moves of the root.
              let mut root_moves = vec![];
              mcts.get_state().move_gen::<false>(&mut root_moves);
              if !root_moves.contains(&game_move) {
                panic!("move {} not in root moves: {:?}", game_move, root_moves);
              }
              //println!("[{task_id}] Generated a move: {:?}", m);
              was_rand.push(pick_randomly);
              used_mate_move.push(mating_move.is_some());
              mate_search_evals.push(alpha_beta_eval);
              bad_non_bad_moves.push((bad_move, non_bad_move));
              alpha_beta_search_depths.push(alpha_beta_depth);
              full_search.push(do_full_search);
              train_dists.push(match (mating_move, do_full_search) {
                (Some(mating_move), _) => Some(vec![(mating_move, 1.0)]),
                (None, true) => Some(mcts.get_train_distribution(true)),
                (None, false) => None,
              });
              visit_dists.push(mcts.get_visit_distribution());
              let (v, u) = mcts.get_root_score();
              root_values.push(v);
              root_visits.push(u);
              steps_performed.push(steps);
              moves.push(game_move);
              mcts.apply_move(game_move);
              mcts.get_state().sanity_check().unwrap();
            }
          }

          // EDIT: Don't adjudicate for lone king vs lone king anymore.
          // TODO: re-examine any situations where I can adjudicate early.
          // // If the game has been lone king vs lone king for three moves, then we're done.
          // if mcts.get_state().is_just_kings() {
          //   just_kings_count += 1;
          // }
          // if just_kings_count >= 3 {
          //   println!("\x1b[94mExiting early to just kings\x1b[0m");
          //   was_adjudicated_early = true;
          //   break;
          // }
        }
        let total_steps = steps_performed.iter().sum::<u32>();
        println!(
          "[{task_id}] Generated a game outcome={:?} len={} total-steps={} steps-per-move={:.2}",
          mcts.get_state().get_outcome(),
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
            "model_start": model_name_start,
            "model_end": inference_engine.get_current_model_name(),
            "outcome": mcts.get_state().get_outcome().map(|o| o.to_str()),
            "was_adjudicated_early": was_adjudicated_early,
            "final_state": mcts.get_state(),
            "moves": moves,
            "states": states,
            "was_rand": was_rand,
            "used_mate_move": used_mate_move,
            "mate_search_eval": mate_search_evals,
            "bad_non_bad_moves": bad_non_bad_moves,
            "alpha_beta_search_depths": alpha_beta_search_depths,
            "train_dists": train_dists,
            "visit_dists": visit_dists,
            "root_values": root_values,
            "root_visits": root_visits,
            "full_search": full_search,
            "steps_performed": steps_performed,
            "playout_cap_randomization_p": PLAYOUT_CAP_RANDOMIZATION_P,
            "full_search_playouts": FULL_SEARCH_PLAYOUTS,
            "small_search_playouts": SMALL_SEARCH_PLAYOUTS,
            "game_len_limit": GAME_LEN_LIMIT,
            "seed": seed,
            "search_params": search_params,
            "is_duck_chess": engine::rules::IS_DUCK_CHESS,
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

  macro_rules! swap_out {
    ($new_model_path:expr, $new_output_dir:expr) => {{
      println!(
        "\x1b[94mLoading new model from {} writing to {}\x1b[0m",
        $new_model_path, $new_output_dir
      );
      match inference_engine.swap_out_model($new_model_path) {
        Ok(_) => {
          println!("\x1b[94mLoaded new model\x1b[0m");
          // We now swap out the output file.
          let mut guard = output_file.lock().await;
          *guard = create_output_file($new_output_dir);
        }
        Err(e) => println!("\x1b[91mFailed to load new model:\x1b[0m {}", e),
      }
    }}
  }

  // Listen to a signals directory, if relevant.
  if let Some(signals_dir) = args.signals_dir {
    loop {
      // Wait three seconds, then check for new files in the signals directory.
      tokio::time::sleep(std::time::Duration::from_secs(3)).await;
      if let Some((new_model_path, new_output_dir)) = check_signals_dir(&signals_dir, &mut previously_processed) {
        swap_out!(&new_model_path, &new_output_dir);
      }
    }
  }

  // Read lines of input from stdin to get instructions.
  let mut stdin = tokio::io::BufReader::new(tokio::io::stdin()).lines();
  loop {
    let line = stdin.next_line().await.unwrap().unwrap();
    let line = line.trim();
    if line.starts_with("swap:::") {
      // Split the line at the two :::s.
      let mut parts = line.split(":::");
      parts.next();
      let new_model_path = parts.next().unwrap();
      let new_output_dir = parts.next().unwrap();
      assert!(parts.next().is_none());
      swap_out!(new_model_path, new_output_dir);
    }
    if line == "status" {
      let file = output_file.lock().await;
      //file.flush().unwrap();
      let metadata = file.metadata().unwrap();
      println!("{} bytes written", metadata.len());
    }
  }
}

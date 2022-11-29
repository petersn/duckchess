#![feature(let_chains)]

use std::sync::atomic::AtomicU64;

use clap::Parser;
use engine::inference::InferenceEngine;
use engine::inference_desktop::TensorFlowEngine;
use engine::mcts::{Mcts, PendingPath, SearchParams};
use tokio::sync::Mutex;

const GAME_LEN_LIMIT: usize = 500;
const BATCH_SIZE: usize = 64;

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

  #[arg(long)]
  randomize_search_params: bool,

  #[arg(long)]
  game_limit: Option<u64>,
}

fn generate_random_search_params() -> SearchParams {
  let exploration_alpha = match rand::random::<u32>() % 4 {
    0 => 0.25,
    1 => 0.5,
    2 => 1.0,
    3 => 2.0,
    _ => unreachable!(),
  };
  let duck_exploration_alpha = match rand::random::<u32>() % 4 {
    0 => 0.25,
    1 => 0.5,
    2 => 1.0,
    3 => 2.0,
    _ => unreachable!(),
  };
  let first_play_urgency = match rand::random::<u32>() % 4 {
    0 => 0.0,
    1 => 0.1,
    2 => 0.2,
    3 => 0.4,
    _ => unreachable!(),
  };
  SearchParams {
    exploration_alpha,
    duck_exploration_alpha,
    first_play_urgency,
  }
}

#[tokio::main]
async fn main() {
  let args = Args::parse();

  let model1_dir: &'static str = Box::leak(String::into_boxed_str(args.model1_dir));
  let model2_dir: &'static str = Box::leak(String::into_boxed_str(args.model2_dir));

  let inference_engine1: &TensorFlowEngine<(usize, PendingPath)> =
    Box::leak(Box::new(TensorFlowEngine::new(BATCH_SIZE, model1_dir)));
  let inference_engine2: &TensorFlowEngine<(usize, PendingPath)> =
    Box::leak(Box::new(TensorFlowEngine::new(BATCH_SIZE, model2_dir)));

  let randomize_search_params: &'static _ = Box::leak(Box::new(args.randomize_search_params));
  //let search_params1: &'static SearchParams = Box::leak(Box::new(args.search_params1));
  //let search_params2: &'static SearchParams = Box::leak(Box::new(args.search_params2));

  let output_path = format!(
    "{}/games-mcts-{:016x}.json",
    args.output_dir,
    rand::random::<u64>()
  );
  println!("Writing to {}", output_path);
  let output_file: &'static _ = Box::leak(Box::new(Mutex::new(
    std::fs::File::create(output_path).unwrap(),
  )));
  let game_count: &'static _ = Box::leak(Box::new(AtomicU64::new(0)));
  let game_limit: &'static _ = Box::leak(Box::new(args.game_limit));
  //println!("Model 1: {} Search parameters 1: {:?}", model1_dir, search_params1);
  //println!("Model 2: {} Search parameters 2: {:?}", model2_dir, search_params2);
  println!("Starting up with batch size: {}", BATCH_SIZE);

  let (tx_channels, mut rx_channels): (Vec<_>, Vec<_>) =
    (0..4 * BATCH_SIZE).map(|_| tokio::sync::mpsc::unbounded_channel()).unzip();
  let tx_channels: &'static _ = Box::leak(Box::new(tx_channels));

  let mut tasks = Vec::new();

  // Spawn some tasks to do inference.
  //for inference_engine in [inference_engine1, inference_engine2] {
  for j in 0..2 {
    let inference_engine = if j == 0 {
      inference_engine1
    } else {
      inference_engine2
    };
    for _ in 0..1 {
      tasks.push(tokio::spawn(async move {
        let mut last_print = std::time::Instant::now();
        let mut evals_since_last_print = 0;
        loop {
          //println!("[{j}] acquire");
          inference_engine.semaphore.acquire().await.unwrap().forget();
          //println!("[{j}] got it");
          // Check if we should be doing work here.
          inference_engine.predict(|inference_results| {
            evals_since_last_print += inference_results.cookies.len();
            for (i, (cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
              //let node_id = inference_engine.node_id(i);
              //let node = mcts.node_mut(node_id);
              //node.inference_result = Some(inference_result.clone());
              //notifiers[node_id].notify_waiters();
              //println!("[{j}] send {}", *cookie);
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
  }

  // Spawn several tasks to run MCTS in parallel.
  let mut rx_pairs = vec![];
  rx_channels.reverse();
  while !rx_channels.is_empty() {
    rx_pairs.push((rx_channels.pop().unwrap(), rx_channels.pop().unwrap()));
  }
  for (task_id, (mut rx1, mut rx2)) in rx_pairs.into_iter().enumerate() {
    tasks.push(tokio::spawn(async move {
      use std::io::Write;
      let mut engine1_is_white = rand::random::<bool>();
      loop {
        // Alternate who starts as white.
        engine1_is_white = !engine1_is_white;
        let seed1 = rand::random::<u64>();
        let seed2 = rand::random::<u64>();
        let (search_params1, search_params2) = match randomize_search_params {
          false => (SearchParams::default(), SearchParams::default()),
          true => {
            // Pick two random search parameter sets.
            let search_params1 = generate_random_search_params();
            // Generate search_params2 until we get an unequal set.
            let mut search_params2 = search_params1.clone();
            while search_params1 == search_params2 {
              search_params2 = generate_random_search_params();
            }
            (search_params1, search_params2)
          }
        };
        let mcts1 = Mcts::new(
          2 * task_id + 0,
          seed1,
          inference_engine1,
          search_params1.clone(),
        );
        let mcts2 = Mcts::new(
          2 * task_id + 1,
          seed2,
          inference_engine2,
          search_params2.clone(),
        );
        let (white_engine_name, black_engine_name) = match engine1_is_white {
          true => (model1_dir, model2_dir),
          false => (model2_dir, model1_dir),
        };
        let (white_search_params, black_search_params) = match engine1_is_white {
          true => (search_params1, search_params2),
          false => (search_params2, search_params1),
        };
        let mut mctses = match engine1_is_white {
          true => [mcts1, mcts2],
          false => [mcts2, mcts1],
        };
        let mut rxes = match engine1_is_white {
          true => [&mut rx1, &mut rx2],
          false => [&mut rx2, &mut rx1],
        };

        // Make sure we fully flush here.
        for (mcts, rx) in mctses.iter_mut().zip(rxes.iter_mut()) {
          while mcts.any_in_flight() {
            //println!("(({task_id})) 2 wait...");
            let (pending_path, model_outputs) = rx.recv().await.unwrap();
            //println!("(({task_id})) 2 got it");
            mcts.process_path(pending_path, model_outputs);
          }
        }

        // This array tracks the moves that actually occur in the game.
        let mut moves: Vec<engine::rules::Move> = vec![];
        // This array tracks how many steps were actually performed to compute this move.
        let mut steps_performed: Vec<u32> = vec![];

        for move_number in 0..GAME_LEN_LIMIT {
          // Getting this right is critical, or we don't see any progress.
          let index = match engine::rules::IS_DUCK_CHESS {
            true => (move_number / 2) % 2,
            false => move_number % 2,
          };
          let mcts = &mut mctses[index];
          let rx = &mut rxes[index];

          // Perform the actual tree search.
          // We early out only if we're not doing a full search, to properly compute our training target.
          let mut steps = 0;
          for _ in 0..args.playouts {
            if mcts.have_reached_visit_count_short_circuiting(args.playouts) {
              break;
            }
            steps += 1;
            mcts.step();
            // Make sure we fully flush here.
            while mcts.any_in_flight() {
              //println!("(({task_id})) 1 wait...");
              let (pending_path, model_outputs) = rx.recv().await.unwrap();
              //println!("(({task_id})) 1 got it");
              mcts.process_path(pending_path, model_outputs);
            }
          }
          let game_move = mcts.sample_move_by_visit_count(1);
          match game_move {
            None => break,
            Some(game_move) => {
              steps_performed.push(steps);
              moves.push(game_move);
              mctses[0].apply_move(game_move);
              mctses[1].apply_move(game_move);
            }
          }
          // Make sure we fully flush here.
          for (mcts, rx) in mctses.iter_mut().zip(rxes.iter_mut()) {
            while mcts.any_in_flight() {
              //println!("(({task_id})) 2 wait...");
              let (pending_path, model_outputs) = rx.recv().await.unwrap();
              //println!("(({task_id})) 2 got it");
              mcts.process_path(pending_path, model_outputs);
            }
          }
        }
        let total_steps = steps_performed.iter().sum::<u32>();
        println!(
          "[{task_id}] Generated a game outcome={:?} len={} total-steps={} steps-per-move={:.2}",
          mctses[0].get_state().get_outcome(),
          moves.len(),
          total_steps,
          total_steps as f32 / moves.len() as f32
        );
        // Guarantee that all lists are the same length.

        assert_eq!(moves.len(), steps_performed.len());
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mctses[0].get_state().get_outcome().map(|o| o.to_str()),
            "final_state": mctses[0].get_state(),
            "moves": moves,
            "steps_performed": steps_performed,
            "engine_white": white_engine_name,
            "engine_black": black_engine_name,
            "playouts_white": args.playouts,
            "playouts_black": args.playouts,
            "game_len_limit": GAME_LEN_LIMIT,
            "seed1": seed1,
            "seed2": seed2,
            "search_params_white": white_search_params,
            "search_params_black": black_search_params,
            "is_duck_chess": engine::rules::IS_DUCK_CHESS,
            "version": "mcts-1",
          });
          let s = serde_json::to_string(&obj).unwrap();
          file.write_all(s.as_bytes()).unwrap();
          file.write_all(b"\n").unwrap();
          file.flush().unwrap();
          let gc = 1 + game_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
          if let Some(gl) = *game_limit && gc >= gl {
            println!("Reached game limit of {}", gl);
            std::process::exit(0);
          }
        }
      }
    }));
  }
  // Join all the tasks.
  for task in tasks {
    task.await.unwrap();
  }
  println!("Done!");
}

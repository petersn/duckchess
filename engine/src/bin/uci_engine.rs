use std::{collections::HashMap, io::BufRead, cell::RefCell};

use clap::Parser;
use engine::{inference_desktop::TensorFlowEngine, mcts::{SearchParams, Mcts}};
use engine::inference::InferenceEngine;
use engine::rules::State;

const MAX_BATCH_SIZE: usize = 32;
const MAX_STEPS_BEFORE_INFERENCE: usize = 40 * 32;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  depth: Option<u16>,

  #[arg(short, long)]
  random: bool,

  #[arg(short, long)]
  nnue: bool,

  #[arg(short, long)]
  model: Option<String>,

  #[arg(short, long, default_value = "100")]
  visits: u32,
}

fn main() {
  let args = Args::parse();
  let stdin = std::io::stdin();
  let options = RefCell::new(HashMap::<String, String>::from([
    ("Hash".to_string(), "16".to_string()),
  ]));

  let make_new_engine = || {
    let seed: u64 = rand::random();
    let hash_mib = options.borrow()["Hash"].parse::<usize>().unwrap();
    engine::search::Engine::new(seed, hash_mib * 1024 * 1024)
  };
  let mut engine = make_new_engine();

  let mut mcts_data = args.model.map(|model| {
    let inference_engine: &'static _ = Box::leak(Box::new(TensorFlowEngine::new(MAX_BATCH_SIZE, &model)));
    let mcts = Mcts::new(0, 0, inference_engine, SearchParams::default());
    (inference_engine, mcts)
  });

  macro_rules! inference {
    () => {
      let (inference_engine, mcts) = mcts_data.as_mut().unwrap();
      inference_engine.predict(|inference_results| {
        for (i, (_cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
          mcts.process_path(pending_path.clone(), inference_results.get(i));
        }
      });
    };
    ($inference_engine:expr, $mcts:expr) => {
      $inference_engine.predict(|inference_results| {
        for (i, (_cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
          $mcts.process_path(pending_path.clone(), inference_results.get(i));
        }
      });
    };
  }
  if mcts_data.is_some() {
    inference!();
  }
  let mut mcts_moves_applied = 0;

  for line in stdin.lock().lines().map(|r| r.unwrap()) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
      continue;
    }
    match tokens[0] {
      "uci" => {
        println!("id name DuckChessZero");
        println!("id author Peter Schmidt-Nielsen");
        println!("");
        println!("option name Hash type spin default 16 min 1 max 1048576");
        println!("uciok");
      }
      "isready" => println!("readyok"),
      "quit" => break,
      "ucinewgame" => {
        println!("info string New game");
        engine = make_new_engine();
      }
      "setoption" => {
        assert_eq!(tokens[1], "name");
        assert_eq!(tokens[3], "value");
        let name = tokens[2];
        let value = tokens[4];
        options.borrow_mut().insert(name.to_string(), value.to_string());
      }
      "position" => {
        assert_eq!(tokens[1], "startpos");
        if mcts_data.is_some() {
          // Assert that the position is a continuation of the previous position.
          let (inference_engine, mcts) = mcts_data.as_mut().unwrap();
          let mut state = State::starting_state();
          assert_eq!(tokens[2], "moves");
          if tokens.len() == 3 {
            // This means that the tkens are just "position startpos moves".
            // In which case we should hopefully be good, unless we're doing a second game.
            // FIXME: Make this work for second games.
            continue;
          }
          println!("TOKENS: {:?}", tokens);
          let moves_wanted = tokens.len() - 3 - mcts_moves_applied;
          // Take all the tokens starting at 3, up to the third to last
          let moves = &tokens[3..tokens.len() - moves_wanted];
          println!("MOVES: {:?} (wanted: {})", moves, moves_wanted);
          for m in moves {
            let m = engine::rules::Move::from_uci(m).unwrap();
            state.apply_move::<false>(m, None).unwrap();
          }
          // Assert that this state is equal to the last state in the MCTS.
          assert_eq!(state.get_transposition_table_hash(), mcts.get_state().get_transposition_table_hash());
          // Apply the final moves
          for m in &tokens[tokens.len() - moves_wanted..] {
            let m = engine::rules::Move::from_uci(m).unwrap();
            mcts.apply_move(m);
            mcts_moves_applied += 1;
            inference!(inference_engine, mcts);
          }
        } else {
          engine = make_new_engine();
          if tokens.len() > 2 {
            assert_eq!(tokens[2], "moves");
            let moves = &tokens[3..];
            for m in moves {
              let m = engine::rules::Move::from_uci(m).unwrap();
              engine.apply_move(m).unwrap();
            }
          }
        }
      }
      "go" => {
        let mut depth = 3;
        let mut _infinite = false;
        for i in 1..tokens.len() {
          match tokens[i] {
            "depth" => depth = tokens[i + 1].parse().unwrap(),
            "infinite" => _infinite = true,
            _ => (),
          }
        }
        if let Some(d) = args.depth {
          depth = d;
        }
        //let mut search = engine::search::Search::new(&engine, depth, time, nodes, movetime, infinite, ponder);
        let (score, (m1, m2)) = match (args.random, mcts_data.as_mut()) {
          (true, _) => {
            let mut rng = rand::thread_rng();
            let mut moves = vec![];
            engine.get_state().move_gen::<false>(&mut moves);
            use rand::seq::SliceRandom;
            moves.shuffle(&mut rng);
            let m1 = moves.first().copied();
            let m2 = None;
            (12345, (m1, m2))
          }
          (false, None) => engine.run(depth, args.nnue),
          (false, Some((ref inference_engine, ref mut mcts))) => {
            let mut steps = args.visits;
            while steps > 0 {
              for i in 0..MAX_STEPS_BEFORE_INFERENCE {
                steps -= 1;
                mcts.step();
                mcts.get_state().sanity_check().unwrap(); // FIXME: Maybe remove this?
                if steps == 0 || inference_engine.batch_ready() {
                  break;
                }
              }
              inference!(inference_engine, mcts);
            }
            (0, (mcts.sample_move_by_visit_count(2), None))
          },
        };
        match m1 {
          Some(m1) => {
            println!("bestmove {}", m1.to_uci());
            match m2 {
              Some(m2) => println!("info score cp {} pv {} {}", score, m1.to_uci(), m2.to_uci()),
              None => println!("info score cp {} pv {}", score, m1.to_uci()),
            }
          }
          None => println!("bestmove 0000"),
        }
      }
      _ => eprintln!("Unknown command: {}", line),
    }
  }
}

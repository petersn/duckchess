use std::{collections::HashMap, io::BufRead, cell::RefCell};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  depth: Option<u16>,

  #[arg(short, long)]
  random: bool,
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
        let (score, (m1, m2)) = match args.random {
          false => engine.run(depth),
          true => {
            let mut rng = rand::thread_rng();
            let mut moves = vec![];
            engine.get_state().move_gen::<false>(&mut moves);
            use rand::seq::SliceRandom;
            moves.shuffle(&mut rng);
            let m1 = moves.first().copied();
            let m2 = None;
            (12345, (m1, m2))
          }
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

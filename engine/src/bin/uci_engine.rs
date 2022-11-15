use std::{collections::HashMap, io::BufRead, cell::RefCell};

fn main() {
  let stdin = std::io::stdin();
  let mut options = RefCell::new(HashMap::<String, String>::from([
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
        let mut infinite = false;
        for i in 1..tokens.len() {
          match tokens[i] {
            "depth" => depth = tokens[i + 1].parse().unwrap(),
            "infinite" => infinite = true,
            _ => (),
          }
        }
        //let mut search = engine::search::Search::new(&engine, depth, time, nodes, movetime, infinite, ponder);
        let (score, (m1, m2)) = engine.run(depth);
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
        engine.apply_move(m1.unwrap()).unwrap();
      }
      _ => eprintln!("Unknown command: {}", line),
    }
  }
}

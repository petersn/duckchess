use std::{collections::HashMap, io::BufRead};

fn main() {
  let stdin = std::io::stdin();
  let mut options = HashMap::new();
  let seed: u64 = rand::random();
  let mut engine = engine::search::Engine::new(seed);

  for line in stdin.lock().lines().map(|r| r.unwrap()) {
    let tokens = line.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() {
      continue;
    }
    match tokens[0] {
      "uci" => {
        println!("id name DuckChessZero");
        println!("id author Peter Schmidt-Nielsen");
        println!("uciok");
      }
      "isready" => println!("readyok"),
      "quit" => break,
      "ucinewgame" => {
        println!("info string New game");
        engine = engine::search::Engine::new(seed);
      }
      "setoption" => {
        assert_eq!(tokens[1], "name");
        assert_eq!(tokens[3], "value");
        let name = tokens[2];
        let value = tokens[4];
        options.insert(name.to_string(), value.to_string());
      }
      "position" => {
        assert_eq!(tokens[1], "startpos");
        engine = engine::search::Engine::new(seed);
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

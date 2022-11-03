/// Parses chess.com's flavor of pgn4 used for encoding duck chess games.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Move {
  pub from: u16,
  pub to: u16,
}

#[derive(Debug, Clone)]
pub struct Pgn4 {
  pub headers: HashMap<String, String>,
  pub moves: Vec<Move>,
}

#[derive(Debug)]
enum Token {
  DotDot,
  Number(u64),
  Other(char),
  EndOfText,
}

fn to_index(xy: (u8, u8)) -> u16 {
  (xy.0 as u16) * 8 + (xy.1 as u16)
}

fn lex_moves(mut s: &str) -> Vec<Token> {
  let mut tokens = Vec::new();
  loop {
    s = s.trim_start();
    let mut it = s.chars();
    let c = match it.next() {
      None => break,
      Some(c) => c,
    };
    // Skip comments.
    if let Some(skip_until) = match c {
      ';' => Some('\n'),
      '{' => Some('}'),
      _ => None,
    } {
      s = &s[s.find(skip_until).unwrap_or(s.len())..];
      continue;
    }
    // Parse DotDot
    if s.starts_with("..") {
      tokens.push(Token::DotDot);
      s = &s["..".len()..];
      continue;
    }
    // Parse Number
    let num_prefix_len = s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len());
    if let Ok(number) = s[..num_prefix_len].parse() {
      tokens.push(Token::Number(number));
      s = &s[num_prefix_len..];
      continue;
    }
    // Parse Other
    tokens.push(Token::Other(c));
    s = it.as_str();
  }
  tokens.push(Token::EndOfText);
  tokens.push(Token::EndOfText);
  tokens
}

pub fn parse_pgn4(mut s: &str) -> Result<Pgn4, String> {
  s = s.trim();

  // Parse headers.
  let mut headers = HashMap::new();
  while !s.is_empty() {
    let (line, new_s) = match s.find('\n') {
      Some(i) => (&s[..i], &s[i + 1..]),
      None => (s, ""),
    };
    let line = line.trim();
    if !line.starts_with("[") {
      break;
    }
    if !line.ends_with("]") {
      return Err("Header line must end with ]".to_string());
    }
    let mut parts = line[1..line.len() - 1].splitn(2, ' ');
    let key = parts.next().unwrap();
    let value = parts.next().unwrap();
    if !value.starts_with('"') || !value.ends_with('"') {
      return Err("Header value must be quoted".to_string());
    }
    let value = &value[1..value.len() - 1];
    headers.insert(key.to_string(), value.to_string());
    s = new_s;
  }

  // Parse moves.
  let tokens = lex_moves(s);
  println!("{:?}", tokens);
  let mut moves = Vec::new();
  let mut i = 0;
  loop {
    // Try to parse various structural components.
    match (&tokens[i], &tokens[i + 1]) {
      // Parse a move number and a dot.
      (Token::Number(_), Token::Other('.')) => {
        if moves.len() % 4 != 0 {
          return Err("Badly placed move number".to_string());
        }
        i += 2;
        continue
      }
      // Parse a double dot.
      (Token::DotDot, _) => {
        if moves.len() % 4 != 2 {
          return Err("Badly placed ..".to_string());
        }
        i += 1;
        continue;
      }
      (Token::Other('R'), Token::EndOfText) => {
        // Resignation.
        break;
      }
      (Token::EndOfText, _) => break,
      _ => {}
    }

    // We now try to parse one ply.
    // Start by skipping over a piece name.
    match tokens[i] {
      Token::Other('K') | Token::Other('Q') | Token::Other('R') | Token::Other('B') | Token::Other('N') => {
        println!("skip piece {:?}", tokens[i]);
        i += 1;
      }
      _ => {}
    }
    macro_rules! parse_square {
      () => {
        match (&tokens[i], &tokens[i + 1]) {
          (Token::Other(file@'d'..='k'), Token::Number(rank@4..=11)) => {
            let file = *file as u8 - b'd';
            let rank = *rank as u8 - 4;
            i += 2;
            (file, rank)
          }
          _ => return Err("Expected square".to_string()),
        }
      }
    }
    macro_rules! expect_char {
      ($c:expr) => {
        match tokens[i] {
          Token::Other($c) => i += 1,
          _ => return Err(format!("Expected {}, got {:?}", $c, tokens[i])),
        }
      }
    }
    let departure_square = parse_square!();
    expect_char!('-');
    let arrival_square = parse_square!();
    println!("{:?}-{:?}", departure_square, arrival_square);
    moves.push(Move {
      from: to_index(departure_square),
      to: to_index(arrival_square),
    });
    // Now we must have a duck move.
    expect_char!('Θ');
    // The departure square is optional.
    let departure_square = match tokens[i] {
      Token::Other('-') => None,
      _ => Some(parse_square!()),
    };
    expect_char!('-');
    let arrival_square = parse_square!();
    let departure_square = departure_square.unwrap_or(arrival_square);
    moves.push(Move {
      from: to_index(departure_square),
      to: to_index(arrival_square),
    });
  }

  Ok(Pgn4 { headers, moves })
}

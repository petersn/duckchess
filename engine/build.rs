// build.rs

use std::env;
use std::fs;
use std::path::Path;

#[path = "src/rng.rs"]
mod rng;

fn main() {
  let knight_moves: Vec<u64> = (0..64)
    .map(|pos| {
      let mut mask = 0;
      let (x, y) = (pos % 8, pos / 8);
      for offset in &[
        (1, 2),
        (1, -2),
        (-1, 2),
        (-1, -2),
        (2, 1),
        (2, -1),
        (-2, 1),
        (-2, -1),
      ] {
        let (x, y) = (x as i32 + offset.0, y as i32 + offset.1);
        if x >= 0 && x < 8 && y >= 0 && y < 8 {
          mask |= 1 << (x + y * 8);
        }
      }
      mask
    })
    .collect();
  let king_moves: Vec<u64> = (0..64)
    .map(|pos| {
      let mut mask = 0;
      let (x, y) = (pos % 8, pos / 8);
      for offset in &[
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
      ] {
        let (x, y) = (x as i32 + offset.0, y as i32 + offset.1);
        if x >= 0 && x < 8 && y >= 0 && y < 8 {
          mask |= 1 << (x + y * 8);
        }
      }
      mask
    })
    .collect();
  let rays: Vec<Vec<u64>> = (0..8)
    .map(|ray_dir| {
      (0..64)
        .map(|pos| {
          let mut mask = 0;
          let (x, y) = (pos % 8, pos / 8);
          let (dx, dy) = match ray_dir {
            0 => (1, 0),
            1 => (1, 1),
            2 => (0, 1),
            3 => (-1, 1),
            4 => (-1, 0),
            5 => (-1, -1),
            6 => (0, -1),
            7 => (1, -1),
            _ => unreachable!(),
          };
          for i in 1..8 {
            let (x, y) = (x as i32 + dx * i, y as i32 + dy * i);
            if x >= 0 && x < 8 && y >= 0 && y < 8 {
              mask |= 1 << (x + y * 8);
            }
          }
          mask
        })
        .collect()
    })
    .collect();
  // We have:
  //   12 side+pieces * 64 squares,
  //   64 duck positions,
  //   64 en passant positions,
  //   4 castling rights,
  //   1 is white's turn,
  //   1 is duck move,
  let zobrist_table_length = 64 * 12 + 64 + 64 + 4 + 1 + 1;
  let rng = rng::Rng::new(0);
  // Use rng.next_random() -> u64 to get random numbers.
  let zobrist_table: Vec<u64> = (0..zobrist_table_length).map(|_| rng.next_random()).collect();
  let code = format!(
    r#"
      pub const KNIGHT_MOVES: [u64; 64] = [{}];
      pub const KING_MOVES: [u64; 64] = [{}];
      pub const RAYS: [[u64; 64]; 8] = [{}];
      pub const ZOBRIST: [u64; {}] = [{}];
    "#,
    knight_moves.iter().map(|x| format!("0x{:016x}", x)).collect::<Vec<_>>().join(", "),
    king_moves.iter().map(|x| format!("0x{:016x}", x)).collect::<Vec<_>>().join(", "),
    rays
      .iter()
      .map(|ray| {
        format!(
          "[{}]",
          ray.iter().map(|x| format!("0x{:016x}", x)).collect::<Vec<_>>().join(", ")
        )
      })
      .collect::<Vec<_>>()
      .join(", "),
    zobrist_table_length,
    zobrist_table.iter().map(|x| format!("0x{:016x}", x)).collect::<Vec<_>>().join(", "),
  );

  let out_dir = env::var_os("OUT_DIR").unwrap();
  let dest_path = Path::new(&out_dir).join("tables.rs");
  fs::write(&dest_path, code).unwrap();
  println!("cargo:rerun-if-changed=build.rs");
}

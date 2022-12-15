use std::fs::File;

use clap::Parser;
use engine::nnue::Nnue;
use engine::rules::{Move, Player, State};
use ndarray::{Array1, Array2};
use ndarray_npy::{NpzReader, NpzWriter};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  // We accept any number of files.
  #[arg(short, long)]
  files: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();

  let mut nnue = Nnue::new(&State::starting_state(), engine::nnue::BUNDLED_NETWORK);
  let mut state = State::starting_state();
  let mut all_values = Vec::new();
  let mut all_policies = Vec::new();
  let mut all_values_for_white = Vec::new();  

  for f in args.files {
    println!("Loading file: {}", f);
    let mut npz = NpzReader::new(File::open(f)?)?;
    // We print all of the keys in this npz.
    let indices: Array2<i32> = npz.by_name("indices.npy")?;
    let meta: Array2<i32> = npz.by_name("meta.npy")?;
    // Print the shapes.
    println!("indices: {:?}", indices.shape());
    println!("meta: {:?}", meta.shape());

    // Loop over rows.
    for i in 0..indices.shape()[0] {
      let indices_row_ndarray = indices.row(i);
      let indices_row = indices_row_ndarray.as_slice().unwrap();
      let meta_row_ndarray = meta.row(i);
      let meta_row = meta_row_ndarray.as_slice().unwrap();
      nnue.compute_from_indices(indices_row);
      // We copy meta data into the state, as a very hacky hack.
      let value_for_white = meta_row[1];
      state.turn = match meta_row[2] {
        0 => Player::Black,
        1 => Player::White,
        _ => panic!("bad player: {}", meta_row[1]),
      };
      state.is_duck_move = meta_row[3] != 0;

      // run the network.
      let eval = nnue.evaluate_value(&state);
      // Flip the eval to be for white.
      let eval = match state.turn {
        Player::Black => -eval,
        Player::White => eval,
      } as f32;
      let network_output = (eval as f32 / (1 << 14) as f32).tanh();
      all_values.push(network_output);

      let mut policy_from = [0i16; 64];
      let mut policy_to = [0i16; 64];
      nnue.evaluate_policy(&state, &mut policy_from, &mut policy_to);
      all_policies.extend_from_slice(&policy_from);
      all_policies.extend_from_slice(&policy_to);

      all_values_for_white.push(value_for_white);
      //break; // FIXME: For now just do one entry.
      //println!("Eval: {:5.2} (value for white: {})", network_output, value_for_white);
    }
  }

  let mut npz = NpzWriter::new(File::create("rust-nnue-predictions.npz")?);
  npz.add_array("values", &Array1::from(all_values))?;
  npz.add_array("policies", &Array1::from(all_policies).into_shape((all_values_for_white.len(), 2, 64))?)?;
  npz.add_array("value_for_white", &Array1::from(all_values_for_white))?;
  npz.finish()?;

  /*
  let mut nnue = Nnue::new(&state, engine::nnue::BUNDLED_NETWORK);
  let moves = [
    "e2e4", "a3a3",
    "e7e5", "a3a4",
    //"d1h5", "a4a3",
    //"g7g6", "a3a4",
    //"e1e2", "a4a3",
    //"g6h5",
    //"f1a6", "a4a3",
    //"b7a6",
  ];
  for uci in moves {
    let m = Move::from_uci(uci).unwrap();
    // Print the move in json.
    println!("{}", serde_json::to_string(&m).unwrap());
    state.apply_move::<true>(m, Some(&mut nnue)).unwrap();
  }
  println!("Initial hash: {:016x}", nnue.get_debugging_hash());
  nnue.dump_state();
  let eval = nnue.evaluate(&state);
  println!("Eval: {}", eval);
  */
  Ok(())
}

/*
use clap::Parser;
use engine::{
  inference::InferenceEngine, inference_desktop::TensorFlowEngine,
  inference_tensorrt::TensorRTEngine, rules::Move,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  keras_model: String,

  #[arg(short, long)]
  tensorrt_model: String,
}

fn main() {
  let args = Args::parse();

  let keras_engine = TensorFlowEngine::new(16, &args.keras_model);
  let tensorrt_engine = TensorRTEngine::new(16, &args.tensorrt_model);

  let state = engine::rules::State::starting_state();
  let mut next_state = state.clone();
  next_state.apply_move::<false>(Move { from: 8, to: 16 }, None).unwrap();

  keras_engine.add_work(&state, 'a');
  keras_engine.add_work(&next_state, 'b');
  tensorrt_engine.add_work(&state, 'a');
  tensorrt_engine.add_work(&next_state, 'b');
  keras_engine.predict(|keras_outputs| {
    tensorrt_engine.predict(|tensorrt_outputs| {
      // Print the cookies.
      println!("Keras: {:?}", keras_outputs.cookies);
      println!("TensorRT: {:?}", tensorrt_outputs.cookies);
      //println!("outputs: {:#?}", tensorrt_outputs);
      for (row1, row2) in keras_outputs.policies.iter().zip(tensorrt_outputs.policies.iter()) {
        for (p1, p2) in row1.iter().zip(row2.iter()) {
          if (p1 - p2).abs() > 1e-2 {
            println!("Mismatch: {} vs {}", p1, p2);
          }
        }
      }
      println!(
        "Keras: {:#?}, TensorRT: {:#?}",
        keras_outputs.values, tensorrt_outputs.values
      );
    });
  });
}
*/

fn main() {
  println!("Hello, world!");
}

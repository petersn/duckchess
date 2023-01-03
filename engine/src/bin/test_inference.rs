/*
use clap::Parser;
use engine::{inference::InferenceEngine, inference_desktop::TensorFlowEngine};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  model: String,
}

fn main() {
  let args = Args::parse();

  let inference_engine = TensorFlowEngine::new(256, &args.model);
  let state = engine::rules::State::starting_state();

  loop {
    // Time featurization.
    let start = std::time::Instant::now();
    loop {
      if inference_engine.add_work(&state, 'a') {
        break;
      }
    }
    let featurization_time = start.elapsed();
    // Time inference.
    let start = std::time::Instant::now();
    inference_engine.predict(|outputs| {
      //println!("outputs: {:#?}", outputs);
      //println!("Cookies: {:?}", outputs.cookies);
    });
    let inference_time = start.elapsed();
    println!(
      "Featurization: {:?}, Inference: {:?}",
      featurization_time, inference_time
    );
  }
}
*/

fn main() {
  println!("Hello, world!");
}

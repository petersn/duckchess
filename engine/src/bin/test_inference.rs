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

  let inference_engine = TensorFlowEngine::new(1, &args.model);
  let state = engine::rules::State::starting_state();

  inference_engine.add_work(&state, 'a');
  inference_engine.predict(|outputs| {
    println!("outputs: {:#?}", outputs);
  });
}

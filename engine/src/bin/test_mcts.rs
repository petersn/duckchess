use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

#[tokio::main]
async fn main() {
  let inference_engine = InferenceEngine::create("/tmp/keras").await;
  let mut mcts = Mcts::create(&inference_engine).await;
  for _ in 0..100 {
    mcts.step().await;
  }
  println!("Done");
  mcts.print_tree();
}

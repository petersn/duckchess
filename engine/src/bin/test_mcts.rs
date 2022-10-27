use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

#[tokio::main]
async fn main() {
  let inference_engine = InferenceEngine::create("/tmp/keras").await;
  let mut mcts = Mcts::create(&inference_engine).await;
  for _ in 0..100 {
    for _ in 0..100 {
      mcts.step().await;
    }
    //mcts.print_tree();
    let best_move = mcts.sample_move_by_visit_count().unwrap();
    println!("====== Making move: {:?}", best_move);
    mcts.apply_move(best_move).await;
  }

  // //mcts.print_tree();
  // for _ in 0..10 {
  //   mcts.step().await;
  // }
}

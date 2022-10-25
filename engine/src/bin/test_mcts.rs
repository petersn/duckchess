#[tokio::main]
async fn main() {
  let mut mcts = engine::mcts::Mcts::create().await;
  mcts.step().await;
  // Time 100 steps:
  let start = std::time::Instant::now();
  for _ in 0..100 {
    mcts.step().await;
  }
  let elapsed = start.elapsed();
  println!("Time elapsed in expensive_function() is: {:?}", elapsed);
  //mcts.print_tree();
}

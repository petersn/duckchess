#[tokio::main]
async fn main() {
  let mut mcts = engine::mcts::Mcts::create().await;
  for _ in 0..10 {
    mcts.step().await;
  }
  mcts.print_tree();
}

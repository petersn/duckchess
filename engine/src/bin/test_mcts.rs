//use engine::desktop_inference::InferenceEngine;
use engine::mcts::Mcts;

fn main() {
  //let inference_engine = InferenceEngine::create("/tmp/keras").await;
  let mut mcts = Mcts::new(0);
  for _ in 0..1 {
    for _ in 0..3 {
      mcts.step();
    }
    mcts.print_tree_root();
    //let best_move = mcts.sample_move_by_visit_count().unwrap();
    //println!("====== Making move: {:?}", best_move);
    //mcts.apply_move(best_move);
  }

  // //mcts.print_tree();
  // for _ in 0..10 {
  //   mcts.step().await;
  // }
}

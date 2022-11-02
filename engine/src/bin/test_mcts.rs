use engine::inference_desktop::TensorFlowEngine;
use engine::mcts::Mcts;

fn main() {
  let inference_engine = TensorFlowEngine::new("/tmp/keras");
  let mut mcts = Mcts::new(0, &inference_engine);
  mcts.predict_now();
  for _ in 0..2 {
    for _ in 0..3 {
      mcts.step();
      mcts.predict_now();
    }
    mcts.print_tree_root();
    //mcts.predict_now();
    //mcts.print_tree_root();
    //let best_move = mcts.sample_move_by_visit_count().unwrap();
    //println!("====== Making move: {:?}", best_move);
    //mcts.apply_move(best_move);
  }

  // //mcts.print_tree();
  // for _ in 0..10 {
  //   mcts.step().await;
  // }
}

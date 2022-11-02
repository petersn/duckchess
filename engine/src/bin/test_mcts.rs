use engine::inference::InferenceEngine;
use engine::inference_desktop::TensorFlowEngine;
use engine::mcts::Mcts;
use engine::rules::Move;

fn main() {
  let inference_engine = TensorFlowEngine::new("/tmp/keras");
  let mut mcts = Mcts::new(0, 0, &inference_engine);

  macro_rules! inference {
    () => {
      mcts.get_state().sanity_check().unwrap();
      inference_engine.predict(|inference_results| {
        for (i, (cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
          mcts.process_path(pending_path.clone(), inference_results.get(i));
        }
      });
    };
  }
  inference!();

  // e4
  mcts.apply_move(Move { from: 12, to: 28 });
  inference!();
  mcts.apply_move(Move { from: 16, to: 16 });
  inference!();

  // a5
  mcts.apply_move(Move { from: 48, to: 40 });
  inference!();
  mcts.apply_move(Move { from: 16, to: 17 });
  inference!();

  // Qf3
  mcts.apply_move(Move { from: 3, to: 21 });
  inference!();
  mcts.apply_move(Move { from: 17, to: 16 });
  inference!();

  // a4
  mcts.apply_move(Move { from: 40, to: 32 });
  inference!();
  mcts.apply_move(Move { from: 16, to: 17 });
  inference!();

  // Qxf7??
  mcts.apply_move(Move { from: 21, to: 53 });
  inference!();
  mcts.apply_move(Move { from: 17, to: 16 });
  inference!();

  for _ in 0..1 {
    for _ in 0..30 {
      mcts.step();
      inference!();
    }
    mcts.print_tree_root();
    //mcts.predict_now();
    //mcts.print_tree_root();
    //let best_move = mcts.sample_move_by_visit_count().unwrap();
    //println!("====== Making move: {:?}", best_move);
    //mcts.apply_move(best_move);
  }
  // Print out the posterior probabilities of the root node.
  let root_node = &mcts.nodes[mcts.root];
  for m in &root_node.moves {
    println!(
      "  {:?} -> {:?}",
      m,
      root_node.posterior(*m),
    );
  }


  // //mcts.print_tree();
  // for _ in 0..10 {
  //   mcts.step().await;
  // }
}

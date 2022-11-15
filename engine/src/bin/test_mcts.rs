use clap::Parser;
use engine::inference::InferenceEngine;
use engine::inference_desktop::TensorFlowEngine;
use engine::mcts::{Mcts, SearchParams};
use engine::rules::Move;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  #[arg(short, long)]
  model: String,
}

fn main() {
  let args = Args::parse();

  let inference_engine = TensorFlowEngine::new(128, &args.model);
  let mut mcts = Mcts::new(0, 0, &inference_engine, SearchParams::default());

  macro_rules! inference {
    () => {
      mcts.get_state().sanity_check().unwrap();
      inference_engine.predict(|inference_results| {
        for (i, (_cookie, pending_path)) in inference_results.cookies.into_iter().enumerate() {
          mcts.process_path(pending_path.clone(), inference_results.get(i));
        }
      });
    };
  }
  inference!();

  //  // e4
  //  mcts.apply_move(Move { from: 12, to: 28 });
  //  inference!();
  //  mcts.apply_move(Move { from: 16, to: 16 });
  //  inference!();
  //
  //  // a5
  //  mcts.apply_move(Move { from: 48, to: 40 });
  //  inference!();
  //  mcts.apply_move(Move { from: 16, to: 17 });
  //  inference!();
  //
  //  // Qf3
  //  mcts.apply_move(Move { from: 3, to: 21 });
  //  inference!();
  //  mcts.apply_move(Move { from: 17, to: 16 });
  //  inference!();
  //
  //  // a4
  //  mcts.apply_move(Move { from: 40, to: 32 });
  //  inference!();
  //  mcts.apply_move(Move { from: 16, to: 17 });
  //  inference!();
  //
  //  // Qxf7??
  //  mcts.apply_move(Move { from: 21, to: 53 });
  //  inference!();
  //  mcts.apply_move(Move { from: 17, to: 16 });
  //  inference!();

  //let s = r#"{"pawns": [[0, 0, 0, 0, 32, 64, 23, 0], [0, 119, 0, 72, 0, 0, 0, 0]], "knights": [[0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]], "bishops": [[0, 0, 0, 0, 0, 0, 0, 4], [32, 0, 0, 0, 0, 0, 0, 0]], "rooks": [[0, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 128]], "queens": [[0, 0, 0, 0, 0, 0, 0, 8], [8, 0, 0, 0, 0, 0, 0, 0]], "kings": [[0, 0, 0, 0, 0, 0, 0, 16], [16, 0, 0, 0, 0, 0, 0, 0]], "ducks": [0, 0, 0, 0, 0, 0, 0, 0], "enPassant": [0, 0, 0, 0, 0, 0, 0, 0], "castlingRights": [{"kingSide": false, "queenSide": true}, {"kingSide": false, "queenSide": true}], "turn": "black", "isDuckMove": false, "moveHistory": [{"from": 47, "to": 63}, {"from": 53, "to": 37}, {"from": 7, "to": 47}, {"from": 63, "to": 47}]}"#;
  //let state: engine::rules::State = serde_json::from_str(s).unwrap();
  let moves = r#"[{"from": 6, "to": 21}, {"from": 54, "to": 46}, {"from": 21, "to": 31}, {"from": 51, "to": 35}, {"from": 1, "to": 18}, {"from": 62, "to": 47}, {"from": 11, "to": 27}, {"from": 57, "to": 51}, {"from": 15, "to": 23}, {"from": 47, "to": 30}, {"from": 23, "to": 30}, {"from": 55, "to": 47}, {"from": 18, "to": 35}, {"from": 51, "to": 45}, {"from": 31, "to": 37}, {"from": 45, "to": 35}, {"from": 37, "to": 47}, {"from": 35, "to": 41}, {"from": 2, "to": 38}, {"from": 61, "to": 47}, {"from": 38, "to": 47}, {"from": 63, "to": 47}, {"from": 7, "to": 47}, {"from": 53, "to": 37}, {"from": 47, "to": 63}]"#;
  let moves: Vec<Move> = serde_json::from_str(moves).unwrap();
  //mcts.set_state(state);
  for m in moves {
    mcts.apply_move(m);
    inference!();
  }
  println!("=================");
  //println!("{:?}", state);
  println!("{:?}", mcts.get_state());

  for _ in 0..1 {
    for _ in 0..200 {
      mcts.step();
      inference!();
    }
    //mcts.print_tree_as_graphviz();
    //mcts.predict_now();
    mcts.print_tree_root();
    //let best_move = mcts.sample_move_by_visit_count().unwrap();
    //println!("====== Making move: {:?}", best_move);
    //mcts.apply_move(best_move);
  }
  // Print out the posterior probabilities of the root node.
  let root_node = &mcts.nodes[mcts.root];
  for m in &root_node.moves {
    println!("  {} -> {:?}", m, root_node.posterior(*m),);
  }

  // //mcts.print_tree();
  // for _ in 0..10 {
  //   mcts.step().await;
  // }
}

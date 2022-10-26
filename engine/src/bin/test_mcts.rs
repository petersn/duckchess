use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

const STEPS_PER_MOVE: usize = 400;

#[tokio::main]
async fn main() {
  let inference_engine: &InferenceEngine = Box::leak(Box::new(InferenceEngine::create().await));

  let output_path = format!("rl-games/games-{:016x}.json", rand::random::<u64>());
  let output_file: &'static _ = Box::leak(Box::new(tokio::sync::Mutex::new(std::fs::File::create(output_path).unwrap())));

  // Spawn several tasks to run MCTS in parallel.
  let mut tasks = Vec::new();
  for task_id in 0..2*engine::inference::BATCH_SIZE - 1 {
    tasks.push(tokio::spawn(async move {
      use std::io::Write;
      loop {
        let mut mcts = Mcts::create(task_id, inference_engine).await;
        let mut moves = vec![];
        for _ in 0..300 {
          for _ in 0..STEPS_PER_MOVE {
            mcts.step().await;
          }
          match mcts.sample_move_by_visit_count() {
            None => break,
            Some(m) => {
              //println!("[{task_id}] Generated a move: {:?}", m);
              moves.push(m);
              mcts.apply_move(m).await;
            }
          }
        }
        println!("[{task_id}] Generated a game len={}", moves.len());
        {
          let mut file = output_file.lock().await;
          let obj = serde_json::json!({
            "outcome": mcts.get_state().get_outcome().to_str(),
            "moves": moves,
            "steps_per_move": STEPS_PER_MOVE,
            "version": 100,
          });
          let s = serde_json::to_string(&obj).unwrap();
          file.write_all(s.as_bytes()).unwrap();
          file.write_all(b"\n").unwrap();
          file.flush().unwrap();
        }
      }
      //println!("Starting task {} (on thread: {:?})", task_id, std::thread::current().id());
      //println!("[{task_id}] Completed init");
      //for step in 0..1_000_000 {
      //  //println!("Task {} step {} (on thread: {:?})", task_id, step, std::thread::current().id());
      //  mcts.step().await;
      //}
      //println!("[{task_id}] Completed steps");
    }));
  }
  // Join all the tasks.
  for task in tasks {
    task.await.unwrap();
  }
  //println!("Time elapsed in expensive_function() is: {:?}", elapsed);
  //mcts.print_tree();
}

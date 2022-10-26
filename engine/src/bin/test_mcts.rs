use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

#[tokio::main]
async fn main() {
  let inference_engine: &InferenceEngine = Box::leak(Box::new(InferenceEngine::create().await));

  // Spawn several tasks to run MCTS in parallel.
  let mut tasks = Vec::new();
  for task_id in 0..2*engine::inference::BATCH_SIZE - 1 {
    tasks.push(tokio::spawn(async move {
      loop {
        let mut mcts = Mcts::create(task_id, inference_engine).await;
        let mut game = vec![];
        for _ in 0..300 {
          for _ in 0..400 {
            mcts.step().await;
          }
          match mcts.sample_move_by_visit_count() {
            None => break,
            Some(m) => {
              //println!("[{task_id}] Generated a move: {:?}", m);
              game.push(m);
              mcts.apply_move(m).await;
            }
          }
        }
        println!("[{task_id}] Generated a game len={}", game.len());
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

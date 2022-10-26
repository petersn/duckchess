use engine::inference::InferenceEngine;
use engine::mcts::Mcts;

#[tokio::main]
async fn main() {
  let inference_engine: &InferenceEngine = Box::leak(Box::new(InferenceEngine::create().await));

  // Spawn several tasks to run MCTS in parallel.
  let mut tasks = Vec::new();
  for task_id in 0..2*engine::inference::BATCH_SIZE {
    tasks.push(tokio::spawn(async move {
      //println!("Starting task {} (on thread: {:?})", task_id, std::thread::current().id());
      let mut mcts = Mcts::create(task_id, inference_engine).await;
      println!("[{task_id}] Completed init");
      for step in 0..10 {
        //println!("Task {} step {} (on thread: {:?})", task_id, step, std::thread::current().id());
        mcts.step().await;
      }
      println!("[{task_id}] Completed steps");
    }));
  }
  // Join all the tasks.
  for task in tasks {
    task.await.unwrap();
  }
  //println!("Time elapsed in expensive_function() is: {:?}", elapsed);
  //mcts.print_tree();
}

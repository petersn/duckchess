
fn main() {
  let mut e = engine::search::Engine::new(1234, 64 * 1024 * 1024);
  let start_time = std::time::Instant::now();
  //let (score, move_pair) = e.run(6, false);
  let (score, move_pair) = e.mate_search(6);
  let elapsed = start_time.elapsed().as_secs_f32();
  println!("Score: {}", score);
  println!("Moves: {:?}", move_pair);
  println!("Elapsed: {:.3}s", elapsed);
  println!("Nodes: {}", e.nodes_searched);
  println!("Nodes/s: {}", e.nodes_searched as f32 / elapsed);
}

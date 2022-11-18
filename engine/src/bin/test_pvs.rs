
fn main() {
  let mut e = engine::search::Engine::new(1234, 64 * 1024 * 1024);
  let start_time = std::time::Instant::now();
  let (score, (m1, m2)) = e.run(5, false);
  let elapsed = start_time.elapsed().as_secs_f32();
  println!("Score: {}", score);
  println!("Move: {:?}", m1);
  println!("Move: {:?}", m2);
  println!("Elapsed: {:.3}s", elapsed);
  println!("Nodes: {}", e.nodes_searched);
  println!("Nodes/s: {}", e.nodes_searched as f32 / elapsed);
}

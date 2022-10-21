
fn main() {
  loop {
    let mut engine = engine::new_engine(0);
    for _ in 0..300 {
      let p = engine::run_internal(&mut engine, 3);
      //println!("{:?}", p);
      if let Some(m) = p.1.0 {
        engine::apply_move_internal(&mut engine, m);
      } else {
        break;
      }
    }
    println!("Game generated"); 
  }
}

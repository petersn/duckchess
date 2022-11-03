
fn main() {
  // Read example.pgn4 into a string.
  let pgn4_str = std::fs::read_to_string("example2.pgn4").unwrap();
  let pgn4 = engine::pgn4_parse::parse(&pgn4_str);
  println!("{:#?}", pgn4);
}

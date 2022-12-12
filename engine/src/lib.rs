#![feature(extern_types)]

pub mod inference;
pub mod mcts;
pub mod nnue;
pub mod pgn4_parse;
pub mod rng;
pub mod rules;
pub mod search;
pub mod eval;

// Build bindings depending on whether we're targeting web or desktop.

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

// Build an inference engine depending on whether we're targeting web or desktop.

#[cfg(target_arch = "wasm32")]
pub mod inference_web;

//#[cfg(not(target_arch = "wasm32"))]
//pub mod inference_desktop;
#[cfg(not(target_arch = "wasm32"))]
pub mod tensorrt;
#[cfg(not(target_arch = "wasm32"))]
pub mod inference_tensorrt;

// Define a log function that can be used in both web and desktop builds.

#[cfg(target_arch = "wasm32")]
pub use web::log;

#[cfg(not(target_arch = "wasm32"))]
fn log(msg: &str) {
  println!("{}", msg);
}

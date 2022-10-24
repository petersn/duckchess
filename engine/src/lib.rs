pub mod rules;
pub mod search;
pub mod mcts;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

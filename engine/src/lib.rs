pub mod inference;
pub mod mcts;
pub mod rules;
pub mod search;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

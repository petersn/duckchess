#![feature(sync_unsafe_cell)]

pub mod inference;
pub mod mcts;
pub mod rng;
pub mod rules;
pub mod search;

// Build bindings depending on whether we're targeting web or desktop.

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

// Build an inference engine depending on whether we're targeting web or desktop.

#[cfg(target_arch = "wasm32")]
pub mod inference_web;

#[cfg(not(target_arch = "wasm32"))]
pub mod inference_desktop;

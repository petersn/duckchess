#![feature(sync_unsafe_cell)]

pub mod mcts;
pub mod rules;
pub mod search;
pub mod rng;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(target_arch = "wasm32")]
pub mod web_inference;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

#[cfg(not(target_arch = "wasm32"))]
pub mod inference;

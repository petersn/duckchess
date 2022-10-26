#![feature(sync_unsafe_cell)]

pub mod rules;
pub mod search;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

#[cfg(not(target_arch = "wasm32"))]
pub mod inference;

#[cfg(not(target_arch = "wasm32"))]
pub mod mcts;

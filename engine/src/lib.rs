#![feature(sync_unsafe_cell)]

pub mod rules;
pub mod search;
pub mod mcts;

#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(target_arch = "wasm32")]
pub mod web_infernece;

#[cfg(not(target_arch = "wasm32"))]
pub mod python;

#[cfg(not(target_arch = "wasm32"))]
pub mod inference;

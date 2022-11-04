use crate::{rules::{State, Move}, inference::Evaluation};

// 6 layes for our pieces, 6 for theirs, 1 for the duck.
const LINEAR_STATE_SIZE: usize = 32;
const FIRST_LAYER_WEIGHTS: &'static [[i32; LINEAR_STATE_SIZE]] = &[[0; LINEAR_STATE_SIZE]; 13 * 64];

pub const DUCK_LAYER_OFFSET: usize = 12 * 64;

pub struct Nnue {
  linear_state: [i32; LINEAR_STATE_SIZE],
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let mut linear_state = [0; LINEAR_STATE_SIZE];
    // Apply the state here.
    Self {
      linear_state,
    }
  }

  pub fn add_layer(&mut self, layer: usize) {

  }

  pub fn sub_layer(&mut self, layer: usize) {

  }

  pub fn sub_add_layers(&mut self, from: usize, to: usize) {
    
  }

  pub fn evaluate(&self) -> Evaluation {
    todo!();
  }
}

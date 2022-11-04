use crate::{
  inference::Evaluation,
  rules::{Move, State},
};

// 6 layes for our pieces, 6 for theirs, 1 for the duck.
const LINEAR_STATE_SIZE: usize = 32;
const SIZE1: usize = 64;
const SIZE2: usize = 1;
const SIZE3: usize = 64;

const LAYER0_WEIGHT: &'static [[i32; LINEAR_STATE_SIZE]] = &[[0; LINEAR_STATE_SIZE]; 13 * 64];
const LAYER0_BIASES: &'static [f32; LINEAR_STATE_SIZE] = &[0.4; LINEAR_STATE_SIZE];

const LAYER_WEIGHTS1: &'static [[f32; SIZE1]; LINEAR_STATE_SIZE] = &[[0.3; SIZE1]; LINEAR_STATE_SIZE];
const LAYER_BIASES1: &'static [f32; SIZE1] = &[0.3; SIZE1];
const LAYER_WEIGHTS2: &'static [[f32; SIZE2]; SIZE1] = &[[0.3; SIZE2]; SIZE1];
const LAYER_BIASES2: &'static [f32; SIZE2] = &[0.3; SIZE2];
const LAYER_WEIGHTS3: &'static [[f32; SIZE3]; SIZE2] = &[[0.3; SIZE3]; SIZE2];
const LAYER_BIASES3: &'static [f32; SIZE3] = &[0.3; SIZE3];

pub const DUCK_LAYER: usize = 12;

pub struct UndoCookie {
  pub sub_layers: [u16; 2],
  pub add_layers: [u16; 2],
}

impl UndoCookie {
  pub fn new() -> UndoCookie {
    UndoCookie {
      sub_layers: [u16::MAX; 2],
      add_layers: [u16::MAX; 2],
    }
  }
}

pub struct Nnue {
  linear_state: [i32; LINEAR_STATE_SIZE],
}

impl Nnue {
  pub fn new(state: &State) -> Self {
    let mut this = Self {
      linear_state: [0; LINEAR_STATE_SIZE],
    };

    this
  }

  pub fn add_layer(&mut self, layer: u16) {
    for (i, weight) in LAYER0_WEIGHT[layer as usize].iter().enumerate() {
      self.linear_state[i] += weight;
    }
  }

  pub fn sub_layer(&mut self, layer: u16) {
    for (i, weight) in LAYER0_WEIGHT[layer as usize].iter().enumerate() {
      self.linear_state[i] -= weight;
    }
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    for (i, weight) in LAYER0_WEIGHT[from as usize].iter().enumerate() {
      self.linear_state[i] -= weight;
    }
    for (i, weight) in LAYER0_WEIGHT[to as usize].iter().enumerate() {
      self.linear_state[i] += weight;
    }
  }

  pub fn undo(&mut self, cookie: UndoCookie) {
    for sub in cookie.sub_layers {
      if sub != u16::MAX {
        // Add because we're undoing a past sub.
        self.add_layer(sub);
      }
    }
    for add in cookie.add_layers {
      if add != u16::MAX {
        // Sub because we're undoing a past add.
        self.sub_layer(add);
      }
    }
  }

  pub fn evaluate(&self) -> Evaluation {
    // Add layer0 bias and apply relu.
    let mut linear_scratch = *LAYER0_BIASES;
    for i in 0..LINEAR_STATE_SIZE {
      linear_scratch[i] = (self.linear_state[i] as f32).max(0.0);
    }
    // Apply layer1 mat mul, bias, and relu.
    let mut layer1_scratch = *LAYER_BIASES1;
    let mut sum = 0.0;
    for i in 0..SIZE1 {
      for j in 0..LINEAR_STATE_SIZE {
        layer1_scratch[i] += linear_scratch[j] * LAYER_WEIGHTS1[j][i];
      }
      layer1_scratch[i] = layer1_scratch[i].max(0.0);
      sum += layer1_scratch[i];
    }
    //// Apply layer2 mat mul, bias, and relu.
    //let mut layer2_scratch = *LAYER_BIASES2;
    //for i in 0..SIZE2 {
    //  for j in 0..SIZE1 {
    //    layer2_scratch[i] += layer1_scratch[j] * LAYER_WEIGHTS2[j][i];
    //  }
    //  layer2_scratch[i] = layer2_scratch[i].max(0.0);
    //}
    //// Apply layer3 mat mul, bias, and relu.
    //let mut layer3_scratch = *LAYER_BIASES3;
    //for i in 0..SIZE3 {
    //  for j in 0..SIZE2 {
    //    layer3_scratch[i] += layer2_scratch[j] * LAYER_WEIGHTS3[j][i];
    //  }
    //  layer3_scratch[i] = layer3_scratch[i].max(0.0);
    //}

    // Return the evaluation.
    Evaluation {
      //expected_score: layer1_scratch[0],
      expected_score: sum,
      perspective_player: crate::rules::Player::White,
    }
  }
}

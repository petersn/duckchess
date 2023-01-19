use std::cell::RefCell;
use std::collections::VecDeque;

use wasm_bindgen::prelude::*;

use crate::inference::{InferenceResults, InputBlock, FEATURES_SIZE, POLICY_LEN};
use crate::{
  inference, mcts,
  rules::{Player, State},
  web,
};

pub const MAX_BATCH_SIZE: usize = 16;

struct PendingResults<Cookie> {
  cookies:  Vec<Cookie>,
  players:  Vec<Player>,
  policies: Vec<[f32; POLICY_LEN]>,
  wdl:      Vec<[f32; 3]>,
}

struct Inner<Cookie> {
  input_blocks:    VecDeque<InputBlock<Cookie>>,
  pending_cookies: Option<Vec<Cookie>>,
  pending_players: Option<Vec<Player>>,
  pending_results: Option<PendingResults<Cookie>>,
}

pub struct TensorFlowJsEngine<Cookie> {
  inner: RefCell<Inner<Cookie>>,
}

impl<Cookie> TensorFlowJsEngine<Cookie> {
  pub fn new() -> Self {
    Self {
      inner: RefCell::new(Inner {
        input_blocks:    VecDeque::new(),
        pending_cookies: None,
        pending_players: None,
        pending_results: None,
      }),
    }
  }

  pub fn has_batch(&self) -> bool {
    let inner = self.inner.borrow_mut();
    inner.input_blocks.front().map(|b| b.cookies.len() == MAX_BATCH_SIZE).unwrap_or(false)
  }

  pub fn fetch_work(&self, array: &mut [f32]) -> usize {
    let mut inner = self.inner.borrow_mut();
    let block = match inner.input_blocks.pop_front() {
      Some(block) => block,
      None => return 0,
    };
    let batch_length = block.cookies.len();
    //web::log(&format!(
    //  "Fetched batch of length {} -> {}",
    //  batch_length,
    //  array.len()
    //));
    // Avoid the call to .subarray if we don't need it.
    if batch_length == MAX_BATCH_SIZE {
      array.copy_from_slice(&block.data[..batch_length * FEATURES_SIZE]);
    } else {
      array[0..batch_length * FEATURES_SIZE]
        //.subarray(0, (batch_length * FEATURES_SIZE) as u32)
        .copy_from_slice(&block.data[..batch_length * FEATURES_SIZE]);
    }
    assert!(inner.pending_cookies.is_none());
    inner.pending_cookies = Some(block.cookies);
    inner.pending_players = Some(block.players);
    batch_length
  }

  //pub fn give_answers(&self, policy_array: &[f32], value_array: &[f32]) {
  pub fn give_answers(&self, policy_array: &[f32], wdl_array: &[f32]) {
    let mut inner = self.inner.borrow_mut();
    let pending_cookies = inner.pending_cookies.take().unwrap();
    let pending_players = inner.pending_players.take().unwrap();
    let batch_length = pending_cookies.len();
    assert!(inner.pending_results.is_none());
    let mut results = PendingResults {
      cookies:  pending_cookies,
      players:  pending_players,
      policies: vec![[0.0; POLICY_LEN]; batch_length], //Box::new([[0.0; POLICY_LEN]; batch_length]),
      wdl:      vec![[0.0; 3]; batch_length],
    };
    // Get a &mut [f32] that points into results.policies.
    let policies = results.policies.as_mut_slice();
    let policies: &mut [f32] = unsafe {
      std::slice::from_raw_parts_mut(policies.as_mut_ptr() as *mut f32, batch_length * POLICY_LEN)
    };
    // Get a &mut [f32] that points into results.wdl.
    let wdl = results.wdl.as_mut_slice();
    let wdl: &mut [f32] =
      unsafe { std::slice::from_raw_parts_mut(wdl.as_mut_ptr() as *mut f32, batch_length * 3) };
    policies.copy_from_slice(&policy_array[..]);
    wdl.copy_from_slice(&wdl_array[..]);
    //policy_array.copy_to(policies);
    //value_array.copy_to(results.values.as_mut_slice());
    inner.pending_results = Some(results);
    //let mut policies = Vec::new();
    //let mut values = Vec::new();
    //let mut index = 0;
    //for cookie in pending_cookies {
    //  policies.push(array.subarray(index, index + inference::POLICY_LEN).to_vec());
    //  index += inference::POLICY_LEN;
    //  values.push(array.get(index));
    //  index += 1;
    //}
    //let results = InferenceResults::new(&pending_cookies, &policies, &values);
  }
}

impl<Cookie> inference::InferenceEngine<Cookie> for TensorFlowJsEngine<Cookie> {
  const DESIRED_BATCH_SIZE: usize = MAX_BATCH_SIZE;

  fn add_work(&self, state: &crate::rules::State, cookie: Cookie) -> bool {
    //web::log(&format!("add_work: {:?}", state));
    let mut inner = self.inner.borrow_mut();
    inference::add_to_input_blocks(MAX_BATCH_SIZE, &mut inner.input_blocks, state, cookie)
  }

  fn predict(&self, use_outputs: impl FnOnce(InferenceResults<Cookie>)) -> usize {
    let mut inner = self.inner.borrow_mut();
    let results = inner.pending_results.take().unwrap();
    let batch_length = results.cookies.len();
    let policy_refs =
      results.policies.iter().map(|p| p.as_ref().try_into().unwrap()).collect::<Vec<_>>();
    use_outputs(InferenceResults::new(
      &results.cookies,
      &results.players,
      &policy_refs[..],
      &results.wdl,
    ));
    batch_length
  }

  fn clear(&self) {
    let mut inner = self.inner.borrow_mut();
    inner.input_blocks.clear();
  }
}

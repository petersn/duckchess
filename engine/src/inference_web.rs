use wasm_bindgen::prelude::*;
use std::cell::RefCell;

use crate::inference::{InferenceResults, InputBlock, PendingIndex, FEATURES_SIZE, POLICY_LEN};
use crate::{inference, mcts, rules::State, web};

pub const MAX_BATCH_SIZE: usize = 32;

struct Results {
  cookies: Vec<PendingIndex>,
  policies: Vec<[f32; POLICY_LEN]>,
  values: Vec<f32>,
}

struct Inner {
  input_blocks: Vec<InputBlock<MAX_BATCH_SIZE>>,
  pending_cookies: Option<Vec<PendingIndex>>,
  pending_results: Option<Results>,
}

pub struct TensorFlowJsEngine {
  inner: RefCell<Inner>,
}

impl TensorFlowJsEngine {
  pub fn new() -> Self {
    Self {
      inner: RefCell::new(Inner {
        input_blocks: Vec::new(),
        pending_cookies: None,
        pending_results: None,
      }),
    }
  }

  pub fn has_batch(&self) -> bool {
    let inner = self.inner.borrow_mut();
    inner.input_blocks.last().map(|b| b.cookies.len() == MAX_BATCH_SIZE).unwrap_or(false)
  }

  pub fn fetch_work(&self, array: &js_sys::Float32Array) -> usize {
    let mut inner = self.inner.borrow_mut();
    match inner.input_blocks.pop() {
      Some(block) => {
        let batch_length = block.cookies.len();
        array.copy_from(&block.data[..batch_length * FEATURES_SIZE]);
        assert!(inner.pending_cookies.is_none());
        inner.pending_cookies = Some(block.cookies);
        batch_length
      }
      None => 0,
    }
  }

  //pub fn give_answers(&self, policy_array: &[f32], value_array: &[f32]) {
    pub fn give_answers(&self, policy_array: &js_sys::Float32Array, value_array: &js_sys::Float32Array) {
    let mut inner = self.inner.borrow_mut();
    let pending_cookies = inner.pending_cookies.take().unwrap();
    let batch_length = pending_cookies.len();
    assert!(inner.pending_results.is_none());
    let mut results = Results {
      cookies: pending_cookies,
      policies: vec![[0.0; POLICY_LEN]; batch_length], //Box::new([[0.0; POLICY_LEN]; batch_length]),
      values: vec![0.0; batch_length],
    };
    // Get a &mut [f32] that points into results.policies.
    let policies = results.policies.as_mut_slice();
    let policies = unsafe {
      std::slice::from_raw_parts_mut(
        policies.as_mut_ptr() as *mut f32,
        batch_length * POLICY_LEN,
      )
    };
    //policies.copy_from_slice(policy_array[..]);
    //results.values.copy_from_slice(value_array[..]);
    policy_array.copy_to(policies);
    value_array.copy_to(results.values.as_mut_slice());
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

impl inference::InferenceEngine for TensorFlowJsEngine {
  const DESIRED_BATCH_SIZE: usize = MAX_BATCH_SIZE;

  fn add_work(&self, state: &crate::rules::State, cookie: PendingIndex) -> usize {
    //web::log(&format!("add_work: {:?}", state));
    let mut inner = self.inner.borrow_mut();
    inference::add_to_input_blocks(&mut inner.input_blocks, state, cookie)
  }

  fn predict(&self, use_outputs: impl FnOnce(InferenceResults)) -> usize {
    let mut inner = self.inner.borrow_mut();
    let results = inner.pending_results.take().unwrap();
    let batch_length = results.cookies.len();
    let policy_refs = results.policies.iter().map(|p| p.as_ref().try_into().unwrap()).collect::<Vec<_>>();
    use_outputs(InferenceResults::new(&results.cookies, &policy_refs[..], &results.values));
    batch_length
  }

  fn clear(&self) {
    let mut inner = self.inner.borrow_mut();
    inner.input_blocks.clear();
  }
}

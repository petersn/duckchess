use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use crate::inference;
use crate::inference::{InferenceResults, InputBlock, CHANNEL_COUNT, FEATURES_SIZE, POLICY_LEN};
use crate::tensorrt::TensorRT;

pub struct TensorRTEngine<Cookie> {
  max_batch_size: usize,
  tensorrt:       TensorRT,
  input_blocks:   Mutex<VecDeque<InputBlock<Cookie>>>,
  pub semaphore:  tokio::sync::Semaphore,
  eval_count:     std::sync::atomic::AtomicUsize,
}

// Regardless of the maximum batch size we care about, the saved TensorRT engine
// always wants buffers big enough to accomodate 128 inputs/outputs.
const SAVED_MODEL_BATCH_SIZE: usize = 128;

impl<Cookie> TensorRTEngine<Cookie> {
  pub fn new(max_batch_size: usize, model_path: &str) -> Self {
    let mut tensorrt = TensorRT::new(0, SAVED_MODEL_BATCH_SIZE as i32);
    tensorrt.load_model(model_path);
    Self {
      max_batch_size,
      input_blocks: Mutex::new(VecDeque::new()),
      tensorrt,
      semaphore: tokio::sync::Semaphore::new(0),
      eval_count: std::sync::atomic::AtomicUsize::new(0),
    }
  }

  pub fn batch_ready(&self) -> bool {
    let input_blocks = self.input_blocks.lock().unwrap();
    input_blocks.front().map(|b| b.cookies.len() == self.max_batch_size).unwrap_or(false)
  }

  pub fn swap_out_model(&self, model_path: &str) -> Result<(), String> {
    self.tensorrt.load_model(model_path);
    Ok(())
  }

  pub fn get_current_model_name(&self) -> String {
    self.tensorrt.get_current_model_name().to_string()
  }
}

impl<Cookie> inference::InferenceEngine<Cookie> for TensorRTEngine<Cookie> {
  // FIXME: This value doesn't really make sense for the TensorRT engine.
  const DESIRED_BATCH_SIZE: usize = 128;

  // FIXME: This should just be a part of the InferenceEngine trait.
  fn add_work(&self, state: &crate::rules::State, cookie: Cookie) -> bool {
    let mut input_blocks = self.input_blocks.lock().unwrap();
    let ready =
      inference::add_to_input_blocks(self.max_batch_size, &mut input_blocks, state, cookie);
    // FIXME: This notifies too much.
    if ready {
      self.semaphore.add_permits(1);
    }
    ready
  }

  fn predict(&self, use_outputs: impl FnOnce(InferenceResults<Cookie>)) -> usize {
    //let overall_start = std::time::Instant::now();
    // Pop the last input block, which is inside the mutex.
    let last_block = {
      let mut guard = self.input_blocks.lock().unwrap();
      match guard.pop_front() {
        Some(block) => block,
        None => return 0,
      }
    };
    let block_len = last_block.cookies.len();
    assert!(block_len <= self.max_batch_size);

    // We now allocate a stream from the TensorRT engine.
    let slot = self.tensorrt.acquire_slot();
    //let (inp_features_ptr, out_wdl_ptr, out_policy_ptr) = {
    //  let guard = self.tensorrt.lock().unwrap();
    //  guard.get_pointers(stream_id)
    //};
    // Copy data into the tensorrt input buffer.
    //let begin_copy = std::time::Instant::now();
    unsafe {
      std::ptr::copy_nonoverlapping(
        last_block.data.as_ptr(),
        slot.inp_features,
        block_len * FEATURES_SIZE,
      );
    }
    //let elapsed = overall_start.elapsed();
    //let elapsed_copy = begin_copy.elapsed();
    //println!("Total: {:?} Copy time: {:?}", elapsed, elapsed_copy);
    // Run the inference.
    //let start_time = std::time::Instant::now();
    self.tensorrt.run_inference(&slot);
    self.tensorrt.wait_for_inference(&slot);

    //tensorrt.run_inference(stream_id);
    //tensorrt.wait_for_inference(stream_id);
    self.eval_count.fetch_add(block_len, std::sync::atomic::Ordering::Relaxed);
    //let elapsed = start_time.elapsed();
    //println!("Inference time: {:?}", elapsed);
    //let final_start = std::time::Instant::now();
    // Copy the output data out of the tensorrt output buffer.
    let mut policies: Vec<&[f32; POLICY_LEN]> = vec![];
    for i in 0..block_len {
      let policy_ptr = unsafe { slot.out_policy.add(i * POLICY_LEN) };
      let policy: &[f32; POLICY_LEN] = unsafe { std::mem::transmute(policy_ptr) };
      // Check normalization right now.
      let sum: f32 = policy.iter().sum();
      if (sum - 1.0).abs() > 1e-3 {
        println!("Policy at {} is not normalized: {}", i, sum);
      }
      //let policy = unsafe { std::slice::from_raw_parts(policy_ptr, POLICY_LEN) };
      policies.push(policy);
    }
    use_outputs(InferenceResults::new(
      &last_block.cookies,
      &last_block.hashes,
      &last_block.players,
      &policies,
      unsafe { std::slice::from_raw_parts(slot.out_wdl as *const [f32; 3], block_len) },
    ));
    drop(slot);
    //let elapsed = final_start.elapsed();
    //println!("Use time: {:?}", elapsed);
    block_len
    // Release this slot back to the TensorRT engine.
  }

  fn clear(&self) {
    self.input_blocks.lock().unwrap().clear();
  }
}

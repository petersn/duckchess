use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;

use tensorflow::{Graph, Operation, SavedModelBundle, SessionOptions, SessionRunArgs, Tensor};

use crate::inference;
use crate::inference::{
  InferenceResults, InputBlock, CHANNEL_COUNT, FEATURES_SIZE, POLICY_LEN,
};

pub struct Model {
  bundle:    SavedModelBundle,
  input_op:  Operation,
  policy_op: Operation,
  wdl_op:  Operation,
}

pub struct TensorFlowEngine<Cookie> {
  max_batch_size: usize,
  model:          Mutex<Arc<Model>>,
  input_blocks:   Mutex<VecDeque<InputBlock<Cookie>>>,
  //input_tensors:   [&'static SyncUnsafeCell<Tensor<f32>>; BUFFER_COUNT],
  //return_channels: tokio::sync::Mutex<ReturnChannels>,
  pub semaphore:  tokio::sync::Semaphore,
  eval_count:     std::sync::atomic::AtomicUsize,
}

fn load_model(model_dir: &str) -> Result<Model, String> {
  let mut graph = Graph::new();
  println!("Loading model from {}", model_dir);
  // Load saved model bundle (session state + meta_graph data)
  let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, model_dir)
    .map_err(|e| format!("Failed to load saved model: {}", e))?;
  // Get signature metadata from the model bundle
  let signature = bundle
    .meta_graph_def()
    .get_signature("serving_default")
    .map_err(|e| format!("Failed to get signature: {}", e))?;
  // We now enumerate all of the devices in the session.
  let devices = bundle.session
    .device_list()
    .map_err(|e| format!("Failed to list devices: {}", e))?;
  println!("Available devices:");
  for device in devices {
    println!("  {} - {}", device.name, device.device_type);
  }
  // Print all inputs.
  println!("Inputs:");
  for (name, tensor_info) in signature.inputs() {
    println!("  {}: {:?}", name, tensor_info);
  }
  // Print all outputs.
  println!("Outputs:");
  for (name, tensor_info) in signature.outputs() {
    println!("  {}: {:?}", name, tensor_info);
  }
  // Get input/output info.
  let input_info = signature
    .get_input("inp_features")
    .map_err(|e| format!("Failed to get input info: {}", e))?;
  let policy_info = signature
    .get_output("out_policy")
    .map_err(|e| format!("Failed to get policy info: {}", e))?;
  let wdl_info = signature
    .get_output("out_wdl")
    .map_err(|e| format!("Failed to get wdl info: {}", e))?;
  // Get input/output ops from graph.
  let input_op = graph
    .operation_by_name_required(&input_info.name().name)
    .map_err(|e| format!("Failed to get input op: {}", e))?;
  let policy_op = graph
    .operation_by_name_required(&policy_info.name().name)
    .map_err(|e| format!("Failed to get policy op: {}", e))?;
  let wdl_op = graph
    .operation_by_name_required(&wdl_info.name().name)
    .map_err(|e| format!("Failed to get wdl op: {}", e))?;
  Ok(Model {
    bundle,
    input_op,
    policy_op,
    wdl_op,
  })
}

impl<Cookie> TensorFlowEngine<Cookie> {
  pub fn new(max_batch_size: usize, model_dir: &str) -> TensorFlowEngine<Cookie> {
    //// Initialize model_dir, input tensor, and an empty graph
    //let input_tensors = (0..BUFFER_COUNT)
    //  .map(|_| {
    //    &*Box::leak(Box::new(SyncUnsafeCell::new(Tensor::new(&[
    //      BATCH_SIZE as u64,
    //      CHANNEL_COUNT as u64,
    //      8,
    //      8,
    //    ]))))
    //  })
    //  .collect::<Vec<_>>();
    //let eval_count: &'static _ = Box::leak(Box::new(std::sync::atomic::AtomicUsize::new(0)));

    TensorFlowEngine {
      max_batch_size,
      model: Mutex::new(Arc::new(load_model(model_dir).unwrap())),
      //input_tensors: input_tensors[..].try_into().unwrap(),
      //return_channels: tokio::sync::Mutex::new(ReturnChannels {
      //  next_buffer: 0,
      //  slot_index:  0,
      //  channels:    (0..BUFFER_COUNT * BATCH_SIZE)
      //    .map(|_| None)
      //    .collect::<Vec<_>>()
      //    .try_into()
      //    .map_err(|_| ())
      //    .unwrap(),
      //}),
      input_blocks: Mutex::new(VecDeque::new()),
      semaphore: tokio::sync::Semaphore::new(0),
      eval_count: std::sync::atomic::AtomicUsize::new(0),
    }
  }

  pub fn batch_ready(&self) -> bool {
    let input_blocks = self.input_blocks.lock().unwrap();
    input_blocks.front().map(|b| b.cookies.len() == self.max_batch_size).unwrap_or(false)
  }

  pub fn swap_out_model(&self, model_dir: &str) -> Result<(), String> {
    let loaded_model = Arc::new(load_model(model_dir)?);
    let mut model = self.model.lock().unwrap();
    *model = loaded_model;
    Ok(())
  }
}

impl<Cookie> inference::InferenceEngine<Cookie> for TensorFlowEngine<Cookie> {
  const DESIRED_BATCH_SIZE: usize = 256;

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
    let model: Arc<Model> = {
      let guard = self.model.lock().unwrap();
      let cloned = guard.clone();
      drop(guard);
      cloned
    };
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
    // Create the input tensor.
    let input_data = &last_block.data[..block_len * FEATURES_SIZE];
    let input_tensor = Tensor::new(&[block_len as u64, CHANNEL_COUNT as u64, 8, 8])
      .with_values(input_data)
      .unwrap();
    // Configure inputs and outputs.
    let mut session_run_args = SessionRunArgs::new();
    session_run_args.add_feed(&model.input_op, 0, &input_tensor);
    let policy_ft = session_run_args.request_fetch(&model.policy_op, 1);
    let wdl_ft = session_run_args.request_fetch(&model.wdl_op, 2);
    // Run the actual network.
    model.bundle.session.run(&mut session_run_args).expect("Can't run session");
    self.eval_count.fetch_add(block_len, std::sync::atomic::Ordering::Relaxed);
    // Fetch outputs.
    let policy_output = session_run_args.fetch::<f32>(policy_ft).expect("Can't fetch output");
    let wdl_output = session_run_args.fetch::<f32>(wdl_ft).expect("Can't fetch output");
    println!("Input shape: {:?}", input_tensor.shape());
    println!("Policy output shape: {:?}", policy_output.shape());
    println!("WDL output shape: {:?}", wdl_output.shape());
    println!("Block len: {}", block_len);
    let mut policies: Vec<&[f32; POLICY_LEN]> = vec![];
    for i in 0..block_len {
      let range = i * POLICY_LEN..(i + 1) * POLICY_LEN;
      let slice: &[f32] = &policy_output[range];
      // Policy should already be normalized.
      let sum: f32 = slice.iter().sum();
      debug_assert!((sum - 1.0).abs() < 1e-2);
      policies.push(slice.try_into().unwrap());
    }
    // Pass the outputs to the callback.
    use_outputs(InferenceResults::new(
      &last_block.cookies,
      &last_block.players,
      &policies,
      unsafe { std::slice::from_raw_parts(wdl_output.as_ptr() as *const [f32; 3], block_len) },
    ));
    block_len
  }

  fn clear(&self) {
    self.input_blocks.lock().unwrap().clear();
  }
}

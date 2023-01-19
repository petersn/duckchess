// Rust wrapper for:
//
// extern "C" TensorRTWrapper* TensorRTWrapper_new(int cuda_device, int max_batch_size);
// extern "C" void TensorRTWrapper_delete(TensorRTWrapper* wrapper);
// extern "C" void TensorRTWrapper_load_model(TensorRTWrapper* wrapper, const char* model_path);
// extern "C" void TensorRTWrapper_get_pointers(
//   TensorRTWrapper* wrapper,
//   int stream_id,
//   float** inp_features,
//   float** out_wdl,
//   float** out_policy,
// );
// extern "C" void TensorRTWrapper_run_inference(TensorRTWrapper* wrapper, int stream_id);
// extern "C" void TensorRTWrapper_wait_for_inference(TensorRTWrapper* wrapper, int stream_id);

use std::ffi::CString;

extern "C" {
  pub type TensorRTWrapper;
  pub fn TensorRTWrapper_new(
    cuda_device: ::std::os::raw::c_int,
    max_batch_size: ::std::os::raw::c_int,
  ) -> *mut TensorRTWrapper;
  pub fn TensorRTWrapper_delete(wrapper: *mut TensorRTWrapper);
  pub fn TensorRTWrapper_load_model(
    wrapper: *mut TensorRTWrapper,
    model_path: *const ::std::os::raw::c_char,
  );
  pub fn TensorRTWrapper_get_pointers(
    wrapper: *mut TensorRTWrapper,
    stream_id: ::std::os::raw::c_int,
    inp_features: *mut *mut ::std::os::raw::c_float,
    out_wdl: *mut *mut ::std::os::raw::c_float,
    out_policy: *mut *mut ::std::os::raw::c_float,
  );
  pub fn TensorRTWrapper_run_inference(
    wrapper: *mut TensorRTWrapper,
    stream_id: ::std::os::raw::c_int,
  );
  pub fn TensorRTWrapper_wait_for_inference(
    wrapper: *mut TensorRTWrapper,
    stream_id: ::std::os::raw::c_int,
  );
}

pub struct TensorRT {
  pointer:            *mut TensorRTWrapper,
  current_model_name: String,
}

impl Drop for TensorRT {
  fn drop(&mut self) {
    unsafe {
      TensorRTWrapper_delete(self.pointer);
    }
  }
}

// FIXME: I need to double check that this actually is Send + Sync.
unsafe impl Send for TensorRT {}
unsafe impl Sync for TensorRT {}

impl TensorRT {
  pub fn new(cuda_device: i32, max_batch_size: i32) -> TensorRT {
    unsafe {
      TensorRT {
        pointer:            TensorRTWrapper_new(cuda_device, max_batch_size),
        current_model_name: "<no model>".to_string(),
      }
    }
  }

  pub fn get_current_model_name(&self) -> &str {
    &self.current_model_name
  }

  pub fn load_model(&mut self, model_path: &str) {
    self.current_model_name = model_path.to_string();
    unsafe {
      let cstr = CString::new(model_path).unwrap();
      TensorRTWrapper_load_model(self.pointer, cstr.as_ptr());
    }
  }

  pub fn get_pointers(&self, stream_id: i32) -> (*mut f32, *mut f32, *mut f32) {
    let mut inp_features: *mut f32 = std::ptr::null_mut();
    let mut out_wdl: *mut f32 = std::ptr::null_mut();
    let mut out_policy: *mut f32 = std::ptr::null_mut();
    unsafe {
      TensorRTWrapper_get_pointers(
        self.pointer,
        stream_id,
        &mut inp_features,
        &mut out_wdl,
        &mut out_policy,
      );
    }
    (inp_features, out_wdl, out_policy)
  }

  pub fn run_inference(&self, stream_id: i32) {
    unsafe {
      TensorRTWrapper_run_inference(self.pointer, stream_id);
    }
  }

  pub fn wait_for_inference(&self, stream_id: i32) {
    unsafe {
      TensorRTWrapper_wait_for_inference(self.pointer, stream_id);
    }
  }
}

pub fn use_tensorrt() {
  let mut tensorrt = TensorRT::new(0, 128);
  tensorrt.load_model("/tmp/trt_out");
  let (inp_features, _out_wdl, _out_policy) = tensorrt.get_pointers(0);
  unsafe {
    *inp_features = 1.0;
  }
  // See how fast we can run inference.
  let start = std::time::Instant::now();
  for _ in 0..1000 {
    tensorrt.run_inference(0);
    tensorrt.wait_for_inference(0);
    //tensorrt.run_inference(1);
    //tensorrt.wait_for_inference(2);
    //tensorrt.run_inference(2);
    //tensorrt.wait_for_inference(0);
  }
  let elapsed = start.elapsed();
  println!("Elapsed: {:?}", elapsed);
}

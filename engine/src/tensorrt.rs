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
use std::sync::{Mutex, MutexGuard, atomic::AtomicI32};

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
  pub fn get_stream_count() -> ::std::os::raw::c_int;
}

#[derive(Debug)]
pub struct StreamPointers {
  stream_id:    i32,
  pub inp_features: *mut f32,
  pub out_wdl:      *mut f32,
  pub out_policy:   *mut f32,
}

pub struct TensorRT {
  ctx_handle:         *mut TensorRTWrapper,
  stream_pointerses:  Vec<Mutex<StreamPointers>>,
  next_stream_id:     AtomicI32,
  current_model_name: Mutex<String>,
}

impl Drop for TensorRT {
  fn drop(&mut self) {
    unsafe {
      TensorRTWrapper_delete(self.ctx_handle);
    }
  }
}

// FIXME: I'm still scared that nvinfer isn't thread-safe enough for this to be okay.
unsafe impl Send for TensorRT {}
unsafe impl Sync for TensorRT {}

impl TensorRT {
  pub fn new(cuda_device: i32, max_batch_size: i32) -> TensorRT {
    let mut stream_pointerses = vec![];
    let ctx_handle = unsafe { TensorRTWrapper_new(cuda_device, max_batch_size) };
    let stream_count = unsafe { get_stream_count() };
    for stream_id in 0..stream_count {
      let mut inp_features: *mut f32 = std::ptr::null_mut();
      let mut out_wdl: *mut f32 = std::ptr::null_mut();
      let mut out_policy: *mut f32 = std::ptr::null_mut();
      unsafe {
        TensorRTWrapper_get_pointers(
          ctx_handle,
          stream_id,
          &mut inp_features,
          &mut out_wdl,
          &mut out_policy,
        );
      }
      stream_pointerses.push(Mutex::new(StreamPointers {
        stream_id,
        inp_features,
        out_wdl,
        out_policy,
      }));
    }
    println!("TensorRT(cuda_device={}, max_batch_size={}): {} streams - {:?}",
      cuda_device, max_batch_size, stream_count, stream_pointerses);
    TensorRT {
      ctx_handle,
      stream_pointerses,
      next_stream_id: AtomicI32::new(0),
      current_model_name: Mutex::new("<no model>".to_string()),
    }
  }

  pub fn get_current_model_name(&self) -> String {
    self.current_model_name.lock().unwrap().clone()
  }

  pub fn load_model(&self, model_path: &str) {
    // To swap out the model we must first own all the stream pointers.
    // It is deadlock-free to sequentially acquire all the locks, as everyone acquires them in this same order.
    let locks = self.stream_pointerses.iter().map(|x| x.lock()).collect::<Vec<_>>();
    *self.current_model_name.lock().unwrap() = model_path.to_string();
    unsafe {
      let cstr = CString::new(model_path).unwrap();
      TensorRTWrapper_load_model(self.ctx_handle, cstr.as_ptr());
      drop(cstr);
    }
    drop(locks);
  }

  pub fn acquire_slot(&self) -> MutexGuard<StreamPointers> {
    let stream_id = self.next_stream_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize % self.stream_pointerses.len();
    self.stream_pointerses[stream_id].lock().unwrap()
  }

  /*
  pub fn get_pointers(&self, stream_id: i32) -> (*mut f32, *mut f32, *mut f32) {
    let mut inp_features: *mut f32 = std::ptr::null_mut();
    let mut out_wdl: *mut f32 = std::ptr::null_mut();
    let mut out_policy: *mut f32 = std::ptr::null_mut();
    unsafe {
      TensorRTWrapper_get_pointers(
        self.ctx_handle,
        stream_id,
        &mut inp_features,
        &mut out_wdl,
        &mut out_policy,
      );
    }
    (inp_features, out_wdl, out_policy)
  }
  */

  pub fn run_inference(&self, slot: &MutexGuard<StreamPointers>) {
    unsafe {
      TensorRTWrapper_run_inference(self.ctx_handle, slot.stream_id);
    }
  }

  pub fn wait_for_inference(&self, slot: &MutexGuard<StreamPointers>) {
    unsafe {
      TensorRTWrapper_wait_for_inference(self.ctx_handle, slot.stream_id);
    }
  }
}

pub fn use_tensorrt() {
  let mut tensorrt = TensorRT::new(0, 128);
  tensorrt.load_model("/tmp/trt_out");
  let mut slot = tensorrt.acquire_slot();
  unsafe {
    *slot.inp_features = 1.0;
  }
  // See how fast we can run inference.
  let start = std::time::Instant::now();
  for _ in 0..1000 {
    tensorrt.run_inference(&slot);
    tensorrt.wait_for_inference(&slot);
    //tensorrt.run_inference(1);
    //tensorrt.wait_for_inference(2);
    //tensorrt.run_inference(2);
    //tensorrt.wait_for_inference(0);
  }
  let elapsed = start.elapsed();
  println!("Elapsed: {:?}", elapsed);
}

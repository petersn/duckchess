/// We have three implementations:
/// 1. A SIMD implementation for x86_64.
/// 2. A SIMD implementation for aarch64.
/// 3. A SIMD implementation for wasm32.
use std::collections::{hash_map::DefaultHasher, HashMap};

use crate::inference::Evaluation;
#[rustfmt::skip]
use crate::rules::{Player, State};

#[repr(C, align(64))]
struct AlignedData<T> {
  data: T,
}

static BUNDLED_NETWORK_DUMMY: AlignedData<[u8; include_bytes!("nnue-data.bin").len()]> = AlignedData { data: *include_bytes!("nnue-data.bin") };

pub static BUNDLED_NETWORK: &'static [u8] = &BUNDLED_NETWORK_DUMMY.data;

#[derive(Debug, serde::Deserialize)]
struct NnueWeightDescription<'a> {
  shape: Vec<usize>,
  dtype: &'a str,
  offset: usize,
  shift: usize,
}

#[derive(Debug, serde::Deserialize)]
struct WeightsMetaData<'a> {
  version: &'a str,
  weights: HashMap<String, NnueWeightDescription<'a>>,
}

struct NnueWeights<'a> {
  metadata: WeightsMetaData<'a>,
  file_contents: &'a [u8],
}

impl<'a> NnueWeights<'a> {
  fn from_file(file_contents: &'a [u8]) -> Self {
    let nnue_header = file_contents
      .iter()
      .position(|&x| x == 0)
      .map(|x| std::str::from_utf8(&file_contents[..x]).unwrap())
      .unwrap();
    let metadata: WeightsMetaData = serde_json::from_str(nnue_header).unwrap();
    assert_eq!(metadata.version, "v1");
    Self {
      metadata,
      file_contents,
    }
  }

  fn get_weight<T>(&self, vec_divide: usize, name: &str, type_check: &str, shape_check: &[usize], shift_check: usize) -> &'a [T] {
    self.metadata.weights.get(name).map(|desc| {
      let offset = desc.offset;
      let raw_len = desc.shape.iter().product::<usize>();
      // Make sure the tensor is divisible into this vector type.
      assert_eq!(raw_len % vec_divide, 0);
      let len = desc.shape.iter().product::<usize>() / vec_divide;
      // Slice the contents.
      let slice = &self.file_contents[offset..offset + len * std::mem::size_of::<T>()];
      assert_eq!(desc.dtype, type_check);
      assert_eq!(desc.shape, shape_check);
      assert_eq!(slice.len(), len * std::mem::size_of::<T>());
      assert_eq!(desc.shift, shift_check);
      unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const T, len) }
    }).expect(name)
  }

  fn get_veci8(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [VecI8] {
    self.get_weight::<VecI8>(VECTOR_LENGTH_I8, name, "i8", shape_check, shift_check)
  }

  fn get_veci16(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [VecI16] {
    self.get_weight::<VecI16>(VECTOR_LENGTH_I16, name, "i16", shape_check, shift_check)
  }

  fn get_veci8_2d<const inner: usize>(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [[VecI8; inner]] {
    self.get_weight::<[VecI8; inner]>(VECTOR_LENGTH_I8 * inner, name, "i8", shape_check, shift_check)
  }

  fn get_veci16_2d<const inner: usize>(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [[VecI16; inner]] {
    self.get_weight::<[VecI16; inner]>(VECTOR_LENGTH_I16 * inner, name, "i16", shape_check, shift_check)
  }

  fn get_i8(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [i8] {
    self.get_weight::<i8>(1, name, "i8", shape_check, shift_check)
  }

  fn get_i16(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [i16] {
    self.get_weight::<i16>(1, name, "i16", shape_check, shift_check)
  }
}

const VECTOR_LENGTH_I8: usize = 16;
const VECTOR_LENGTH_I16: usize = 8;

// Our SIMD interface:
//
//   fn veci16_zeros() -> VecI16;
//     Return the vector of eight zeros.
//
//   fn veci16_{add,sub}(a: VecI16, b: VecI16) -> VecI16;
//     Perform elementwise *non-saturating* addition/subtraction.
//     This function is only used for maintaining the accumulator, and thus
//     we use non-saturating arithmetic, to guarantee invertibility.
//
//   fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8;
//     Saturatingly narrow the sixteen input elements into one vector.
//
//   fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16;
//     This function is used for implementing fast matrix multiplication.
//     The only guarantee is that the saturating horizontal sum of the result
//     will increase by the saturating dot product of a and b.
//
//   fn veci16_horizontal_sat_sum(a: VecI16) -> i16;
//     Returns the saturated sum of all eight elements of the vector.
//
// We now switch on the architecture to define these functions.
#[rustfmt::skip]
cfg_if::cfg_if! {
  if #[cfg(target_arch = "x86_64")] {
    use std::arch::x86_64::*;
    type VecI8 = __m128i;
    type VecI16 = __m128i;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { _mm_setzero_si128() } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { _mm_add_epi16(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { _mm_sub_epi16(a, b) } }

    #[inline(always)]
    fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8 { unsafe { _mm_packs_epi16(a, b) } }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      unsafe { _mm_adds_epi16(accum, _mm_maddubs_epi16(a, b)) }
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      unsafe {
        let zeros = _mm_setzero_si128();
        let a = _mm_hadds_epi16(a, zeros);
        let a = _mm_hadds_epi16(a, zeros);
        let a = _mm_hadds_epi16(a, zeros);
        _mm_cvtsi128_si32(a) as i16
      }
    }
  } else if #[cfg(target_arch = "aarch64")] {
    use std::arch::aarch64::*;
    type VecI8 = int8x16_t;
    type VecI16 = int16x8_t;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { vdupq_n_s16(0) } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { vaddq_s16(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { vsubq_s16(a, b) } }

    #[inline(always)]
    fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8 {
      unsafe { vqmovn_high_s16(vqmovn_s16(a), b) }
    }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      unsafe {
        let low_mul = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let high_mul = vmull_high_s8(a, b);
        vqaddq_s16(accum, vqaddq_s16(low_mul, high_mul))
      }
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      unsafe {
        let a = vpaddlq_s16(a);
        let a = vpaddlq_s32(a);
        let a = vaddvq_s64(a);
        // Narrow to 16 bits saturatingly.
        a.clamp(i16::MIN as i64, i16::MAX as i64) as i16
      }
    }
  } else if #[cfg(target_arch = "wasm32")] {
    use std::arch::wasm32::*;
    type VecI8 = v128;
    type VecI16 = v128;

    #[inline(always)]
    fn veci16_zeros() -> VecI16 { unsafe { i16x8_splat(0) } }

    #[inline(always)]
    fn veci16_add(a: VecI16, b: VecI16) -> VecI16 { unsafe { i16x8_add(a, b) } }

    #[inline(always)]
    fn veci16_sub(a: VecI16, b: VecI16) -> VecI16 { unsafe { i16x8_sub(a, b) } }

    #[inline(always)]
    fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8 {
      unsafe { i8x16_narrow_i16x8(a, b) }
    }

    #[inline(always)]
    fn vec_matmul_sat_fma(accum: VecI16, a: VecI8, b: VecI8) -> VecI16 {
      unsafe {
        let a = i16x8_extmul_low_i8x16(a, b);
        let b = i16x8_extmul_high_i8x16(a, b);
        i16x8_add(accum, i16x8_add(a, b))
      }
    }

    #[inline(always)]
    fn veci16_horizontal_sat_sum(a: VecI16) -> i16 {
      unsafe {
        let zeros = i16x8_splat(0);
        let a = i16x8_add_sat(a, i16x8_shuffle::<4, 5, 6, 7, 0, 1, 2, 3>(a, zeros));
        let a = i16x8_add_sat(a, i16x8_shuffle::<2, 3, 4, 5, 6, 7, 0, 1>(a, zeros));
        let a = i16x8_add_sat(a, i16x8_shuffle::<1, 2, 3, 4, 5, 6, 7, 0>(a, zeros));
        i16x8_extract_lane::<0>(a)
      }
    }
  } else {
    compile_error!("Unsupported architecture.");
  }
}

// We cast weights from &[[i8; inner]; outer] to a &[[VecI8; inner / VECTOR_LENGTH_I8]; outer].
macro_rules! get_weights2d_i8 {
  ($weights:expr, $inner:expr, $outer:expr) => {{
    static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I8, 0);
    unsafe {
      &*($weights as *const [i8; $inner] as *const [[VecI8; $inner / VECTOR_LENGTH_I8]; $outer])
    }
  }};
}

// We cast biases from &[i16; inner] to a &[VecI16; inner / VECTOR_LENGTH_I16].
macro_rules! get_bias1d_i16 {
  ($biases:expr, $inner:expr) => {{
    static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I16, 0);
    unsafe { &*($biases as *const i16 as *const [VecI16; $inner / VECTOR_LENGTH_I16]) }
  }};
}

const LINEAR_STATE_SIZE: usize = 256;
const LINEAR_STATE_VECTOR_COUNT: usize = LINEAR_STATE_SIZE / VECTOR_LENGTH_I16;

// 6 layers for our pieces, 6 for theirs, 1 for the duck.
pub const DUCK_LAYER: usize = 12;

pub type LayerIndex = u16;
pub const NO_LAYER: LayerIndex = LayerIndex::MAX;

#[derive(Clone, Copy, Debug)]
pub struct UndoCookie {
  pub sub_layers: [LayerIndex; 2],
  pub add_layers: [LayerIndex; 2],
}

impl UndoCookie {
  pub fn new() -> UndoCookie {
    UndoCookie {
      sub_layers: [NO_LAYER; 2],
      add_layers: [NO_LAYER; 2],
    }
  }
}

struct NnueNet {
  l0_weights: [[VecI8; 256 / VECTOR_LENGTH_I8]; 16],
  l0_bias: [VecI16; 2],
  l1_weights: [VecI8; 32],
  l1_bias: [VecI16; 4],
  l2_weights: [VecI8; 2],
  l2_bias: i16,
}

pub struct Nnue<'a> {
  layers:       &'a [[VecI16; LINEAR_STATE_VECTOR_COUNT]],
  nets:         [NnueNet; 16],
  pub linear_state: [VecI16; LINEAR_STATE_VECTOR_COUNT],
  //pub outputs:      [i16; 64 + 1],
}

impl<'a> Nnue<'a> {
  pub fn new(state: &State, file_contents: &'a [u8]) -> Self {
    //let linear_state = get_bias1d_i16!(PARAMS_MAIN_EMBED_BIAS, LINEAR_STATE_SIZE);
    let weights = NnueWeights::from_file(file_contents);
    let layers = weights.get_veci16_2d::<LINEAR_STATE_VECTOR_COUNT>("main_embed.weight", &[106496, LINEAR_STATE_SIZE], 7);
    let linear_state = weights.get_veci16("main_bias", &[LINEAR_STATE_SIZE], 7);
    let mut nets = vec![];
    for i in 0..16 {
      let net = NnueNet {
        l0_weights: weights.get_veci8_2d::<16>(&format!("n{i}.0.w"), &[16, 256], 0).try_into().unwrap(),
        l0_bias:    weights.get_veci16(&format!("n{i}.0.b"), &[16], 7).try_into().unwrap(),
        l1_weights: weights.get_veci8(&format!("n{i}.1.w"), &[32, 16], 0).try_into().unwrap(),
        l1_bias:    weights.get_veci16(&format!("n{i}.1.b"), &[32], 7).try_into().unwrap(),
        l2_weights: weights.get_veci8(&format!("n{i}.2.w"), &[1, 32], 0).try_into().unwrap(),
        l2_bias:    weights.get_i16(&format!("n{i}.2.b"), &[1], 7)[0],
      };
      nets.push(net);
    }

    let mut this = Self {
      layers,
      nets: nets.try_into().map_err(|_| "Wrong number of nets").unwrap(),
      linear_state: linear_state.try_into().unwrap(),
    };

    for turn in [Player::Black, Player::White] {
      for (piece_layer_number, piece_array) in [
        &state.pawns,
        &state.knights,
        &state.bishops,
        &state.rooks,
        &state.queens,
        &state.kings,
      ]
      .into_iter()
      .enumerate()
      {
        let layer_number = piece_layer_number + 6 * turn as usize;
        let mut bitboard = piece_array[turn as usize].0;
        // Get all of the pieces.
        while let Some(pos) = crate::rules::iter_bits(&mut bitboard) {
          this.add_layer((64 * layer_number + pos as usize) as u16);
        }
      }
    }
    this
  }

  pub fn get_debugging_hash(&self) -> u64 {
    use std::hash::{Hash, Hasher};
    // We cast our linear state to a u8 array.
    let linear_state_u8 = unsafe {
      std::slice::from_raw_parts(
        self.linear_state.as_ptr() as *const u8,
        self.linear_state.len() * std::mem::size_of::<VecI16>(),
      )
    };
    let mut hasher = DefaultHasher::new();
    linear_state_u8.hash(&mut hasher);
    hasher.finish()
  }

  pub fn add_layer(&mut self, layer: u16) {
    for i in 0..LINEAR_STATE_VECTOR_COUNT {
      self.linear_state[i] = veci16_add(self.linear_state[i], self.layers[layer as usize][i]);
    }
  }

  pub fn sub_layer(&mut self, layer: u16) {
    for i in 0..LINEAR_STATE_VECTOR_COUNT {
      self.linear_state[i] = veci16_sub(self.linear_state[i], self.layers[layer as usize][i]);
    }
  }

  pub fn sub_add_layers(&mut self, from: u16, to: u16) {
    for i in 0..LINEAR_STATE_VECTOR_COUNT {
      self.linear_state[i] = veci16_sub(self.linear_state[i], self.layers[from as usize][i]);
      self.linear_state[i] = veci16_add(self.linear_state[i], self.layers[to as usize][i]);
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

  pub fn evaluate(&mut self, state: &State) -> Evaluation {
    if let Some(terminal_eval) = Evaluation::from_terminal_state(state) {
      return terminal_eval;
    }

    // Compute the game phase.
    //let game_phase = state.get_occupied().count_ones() / 8;
    // FIXME: For now just always use phase 3.
    let game_phase = 3;
    let network_index = state.turn as usize + 2 * state.is_duck_move as usize + 4 * game_phase;
    let net = &self.nets[network_index];

    /*
    // The horizontal sum of all of the entries in accum[i] represents the output of layer 0.
    let mut accums0 = [veci16_zeros(); 16];
    let weights0 = get_weights2d_i8!(mat0, 256, 16);
    for i in 0..LINEAR_STATE_VECTOR_COUNT / 2 {
      let v0: VecI16 = self.linear_state[2 * i + 0];
      let v1: VecI16 = self.linear_state[2 * i + 1];
      let v = vec_narrow_pair(v0, v1);
      for j in 0..16 {
        let block = weights0[j][i];
        accums0[j] = vec_matmul_sat_fma(accums0[j], v, block);
      }
    }
    // We now horizontally sum the accums0, add the bias, and apply the activation function.
    let mut layer0: [i8; 16] = [0; 16];
    for i in 0..16 {
      let intermediate = veci16_horizontal_sat_sum(accums0[i]) + bias0[i];
      layer0[i] = (intermediate / 128).clamp(-128, 127) as i8;
    }
    let layer0_simd: VecI8 = unsafe { std::mem::transmute(layer0) };

    // Second layer
    let mut accums1 = [veci16_zeros(); 32];
    let weights1 = get_weights2d_i8!(mat1, 16, 32);
    for j in 0..32 {
      let block = weights1[j][0];
      accums1[j] = vec_matmul_sat_fma(accums1[j], layer0_simd, block);
    }
    let mut layer1: [i8; 32] = [0; 32];
    for i in 0..32 {
      let intermediate = veci16_horizontal_sat_sum(accums1[i]) + bias1[i];
      layer1[i] = (intermediate / 128).clamp(-128, 127) as i8;
    }
    let layer1_simd: [VecI8; 2] = unsafe { std::mem::transmute(layer1) };

    // Final layer
    let weights2 = get_weights2d_i8!(mat2, 32, 1);
    let mut final_accum = veci16_zeros();
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[0], weights2[0][0]);
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[1], weights2[0][1]);
    let final_accum = veci16_horizontal_sat_sum(final_accum) + bias2[0];

    let network_output = (final_accum as f32 / 1024.0).tanh();
    //self.value = (self.outputs[64].tanh() * 1000.0) as i32;
    Evaluation {
      expected_score: (network_output + 1.0) / 2.0,
      perspective_player: state.turn,
    }
    */
    todo!()
  }
}

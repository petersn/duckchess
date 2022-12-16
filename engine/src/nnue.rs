/// We have three implementations:
/// 1. A SIMD implementation for x86_64.
/// 2. A SIMD implementation for aarch64.
/// 3. A SIMD implementation for wasm32.
use std::collections::{hash_map::DefaultHasher, HashMap};

#[rustfmt::skip]
use crate::rules::{Player, PieceKind, Square, State};
use crate::search::{eval_terminal_state, IntEvaluation};

#[repr(C, align(32))]
struct AlignedData<T> {
  data: T,
}

static BUNDLED_NETWORK_DUMMY: AlignedData<[u8; include_bytes!("nnue-data.bin").len()]> =
  AlignedData {
    data: *include_bytes!("nnue-data.bin"),
  };

pub static BUNDLED_NETWORK: &'static [u8] = &BUNDLED_NETWORK_DUMMY.data;

#[derive(Debug, serde::Deserialize)]
struct NnueWeightDescription<'a> {
  shape:  Vec<usize>,
  dtype:  &'a str,
  offset: usize,
  shift:  usize,
}

#[derive(Debug, serde::Deserialize)]
struct WeightsMetaData<'a> {
  version: &'a str,
  weights: HashMap<String, NnueWeightDescription<'a>>,
}

struct NnueWeights<'a> {
  metadata:      WeightsMetaData<'a>,
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

  fn get_weight<T>(
    &self,
    vec_divide: usize,
    name: &str,
    type_check: &str,
    shape_check: &[usize],
    shift_check: usize,
  ) -> &'a [T] {
    self
      .metadata
      .weights
      .get(name)
      .map(|desc| {
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
      })
      .expect(name)
  }

  fn get_veci8(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [VecI8] {
    self.get_weight::<VecI8>(VECTOR_LENGTH_I8, name, "i8", shape_check, shift_check)
  }

  fn get_veci16(&self, name: &str, shape_check: &[usize], shift_check: usize) -> &'a [VecI16] {
    self.get_weight::<VecI16>(VECTOR_LENGTH_I16, name, "i16", shape_check, shift_check)
  }

  fn get_veci8_2d<const INNER: usize>(
    &self,
    name: &str,
    shape_check: &[usize],
    shift_check: usize,
  ) -> &'a [[VecI8; INNER]] {
    self.get_weight::<[VecI8; INNER]>(
      VECTOR_LENGTH_I8 * INNER,
      name,
      "i8",
      shape_check,
      shift_check,
    )
  }

  fn get_veci16_2d<const INNER: usize>(
    &self,
    name: &str,
    shape_check: &[usize],
    shift_check: usize,
  ) -> &'a [[VecI16; INNER]] {
    self.get_weight::<[VecI16; INNER]>(
      VECTOR_LENGTH_I16 * INNER,
      name,
      "i16",
      shape_check,
      shift_check,
    )
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
    fn veci16_shr_bias_to_main(a: VecI16) -> VecI16 { unsafe { _mm_srai_epi16(a, 1) } }

    #[inline(always)]
    fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8 {
      unsafe {
        let x = _mm_packs_epi16(a, b);
        // max with zero to saturate.
        _mm_max_epi8(x, _mm_setzero_si128())
      }
    }

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
    fn veci16_shr_bias_to_main(a: VecI16) -> VecI16 { unsafe { i16x8_shr(a, 4) } }

    #[inline(always)]
    fn vec_narrow_pair(a: VecI16, b: VecI16) -> VecI8 {
      unsafe { i8x16_max(i8x16_splat(0), i8x16_narrow_i16x8(a, b)) }
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

// // We cast weights from &[[i8; inner]; outer] to a &[[VecI8; inner / VECTOR_LENGTH_I8]; outer].
// macro_rules! get_weights2d_i8 {
//   ($weights:expr, $inner:expr, $outer:expr) => {{
//     static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I8, 0);
//     unsafe {
//       &*($weights as *const [i8; $inner] as *const [[VecI8; $inner / VECTOR_LENGTH_I8]; $outer])
//     }
//   }};
// }

// // We cast biases from &[i16; inner] to a &[VecI16; inner / VECTOR_LENGTH_I16].
// macro_rules! get_bias1d_i16 {
//   ($biases:expr, $inner:expr) => {{
//     static_assertions::const_assert_eq!($inner % VECTOR_LENGTH_I16, 0);
//     unsafe { &*($biases as *const i16 as *const [VecI16; $inner / VECTOR_LENGTH_I16]) }
//   }};
// }

const LINEAR_STATE_SIZE: usize = 256;
const LINEAR_STATE_VECTOR_COUNT: usize = LINEAR_STATE_SIZE / VECTOR_LENGTH_I16;

// We number the pieces:
//   black: pawn = 0, knight = 1, bishop = 2, rook = 3, queen = 4, king = 5,
//   white: pawn = 6, knight = 7, bishop = 8, rook = 9, queen = 10, king = 11.
//   duck = 12
// Let the black king's square be BK, and the white king's quare be WK.
// A piece p on square s contributes the following two features:
//   1) 13*64*BK + 64*p + s
//   2) 13*64*WK + 64*p + s + 13*64*64
//
// Every non-king move affects at most four layers.
//
// In other words, our layers are:
// Black king is on square 0:
//       0 -- Black pawn on square 0 (impossible ununsed layer)
//       1 -- Black pawn on square 1
//            ...
//      63 -- Black pawn on square 63
//      64 -- Black knight on square 0
//            ...
//     128 -- Black bishop on square 0
//            ...
//     192 -- Black rook on square 0
//            ...
//     256 -- Black queen on square 0
//            ...
//     320 -- Black king on square 0
//     321 -- Black king on square 1 (impossible unused layer)
//            ...
//     384 -- White pawn on square 0 (impossible unused layer)
//            ...
//     768 -- Duck on square 0
//            ...
// Black king is on square 1:
//     832 -- Black pawn on square 0
//     833 -- Black pawn on square 1 (impossible unused layer)
//            ...
// White king is on square 0:
//   53248 -- Black pawn on square 0 (impossible unused layer)
//            ...
//  106495 -- Duck on square 63

pub type LayerIndex = u32;
pub type ZobristLayerIndex = u16;

#[derive(Debug)]
pub struct PlayerPieceSquare {
  pub player:     Player,
  pub piece_kind: PieceKind,
  pub square:     Square,
}

impl PlayerPieceSquare {
  pub fn get_layers(&self, king_positions: (Square, Square)) -> (LayerIndex, LayerIndex) {
    let (black_king, white_king) = king_positions;
    let player_piece_offset = match self.piece_kind {
      PieceKind::Duck => 12,
      _ => (self.piece_kind as LayerIndex) + 6 * (self.player as LayerIndex),
    };
    let layer_offset = 64 * player_piece_offset + self.square as LayerIndex;
    (
      layer_offset + 13 * 64 * (black_king as LayerIndex),
      layer_offset + 13 * 64 * (white_king as LayerIndex) + 13 * 64 * 64,
    )
  }

  pub fn get_zobrist_layer(&self) -> ZobristLayerIndex {
    let player_piece_offset = match self.piece_kind {
      PieceKind::Duck => 12,
      _ => (self.piece_kind as ZobristLayerIndex) + 6 * (self.player as ZobristLayerIndex),
    };
    player_piece_offset * 64 + self.square as ZobristLayerIndex
  }
}

#[inline]
fn get_king_positions(state: &State) -> (Square, Square) {
  (
    crate::rules::get_square(state.kings[Player::Black as usize].0),
    crate::rules::get_square(state.kings[Player::White as usize].0),
  )
}

#[inline]
fn compute_state_layers(state: &State, mut callback: impl FnMut(LayerIndex)) -> Result<(), ()> {
  // Check if the state is terminal, and if so, refuse to featurize.
  if state.get_outcome().is_some() {
    return Err(());
  }

  let king_positions = get_king_positions(state);

  for player in [Player::Black, Player::White] {
    for (piece_kind, piece_array) in [
      (PieceKind::Pawn, &state.pawns),
      (PieceKind::Knight, &state.knights),
      (PieceKind::Bishop, &state.bishops),
      (PieceKind::Rook, &state.rooks),
      (PieceKind::Queen, &state.queens),
      (PieceKind::King, &state.kings),
    ] {
      let mut bitboard = piece_array[player as usize].0;
      // Get all of the pieces.
      while let Some(square) = crate::rules::iter_bits(&mut bitboard) {
        let (black_layer, white_layer) = PlayerPieceSquare {
          player,
          piece_kind,
          square,
        }
        .get_layers(king_positions);
        callback(black_layer);
        callback(white_layer);
      }
    }
  }
  // Include the duck.
  let mut bitboard = state.ducks.0;
  while let Some(pos) = crate::rules::iter_bits(&mut bitboard) {
    let (black_layer, white_layer) = PlayerPieceSquare {
      player:     Player::Black,
      piece_kind: PieceKind::Duck,
      square:     pos,
    }
    .get_layers(king_positions);
    callback(black_layer);
    callback(white_layer);
  }

  Ok(())
}

fn get_net_index(state: &State, layer_count: i32) -> usize {
  assert!(0 <= layer_count);
  assert!(layer_count <= 66);
  assert!(layer_count % 2 == 0);
  // layer_count will be 2 * piece_count, including the duck.
  // We compute a game phase based on how many pieces are left.
  // Specifically, we compute:
  //    2,  3,  4,  5 pieces left (including duck) -> game_phase = 0
  //    6,  7,  8,  9 pieces left (including duck) -> game_phase = 1
  //        ...
  //   30, 31, 32, 33 pieces left (including duck) -> game_phase = 7
  let game_phase = (layer_count - 3).max(0) / 8;
  let net_index = 4 * game_phase as usize + 2 * (state.turn as usize) + state.is_duck_move as usize;
  assert!(net_index < 32);
  net_index
}

pub fn get_state_layers_and_net_index(state: &State) -> Option<(Vec<LayerIndex>, usize)> {
  let mut layers = Vec::new();
  compute_state_layers(state, |layer| layers.push(layer)).ok()?;
  let layer_count = layers.len() as i32;
  Some((layers, get_net_index(state, layer_count)))
}

#[derive(Debug)]
pub enum NnueAdjustment {
  DuckCreation {
    to: PlayerPieceSquare,
  },
  Normal {
    from:    PlayerPieceSquare,
    to:      PlayerPieceSquare,
    // The capture needn't be on the same square as to due to en passant.
    capture: Option<PlayerPieceSquare>,
  },
  KingInvolved,
}

struct NnueValueNet {
  l0_weights: [[VecI8; 256 / VECTOR_LENGTH_I8]; 16],
  l0_bias:    [i16; 16],
  l1_weights: [VecI8; 32],
  l1_bias:    [i16; 32],
  l2_weights: [VecI8; 2],
  l2_bias:    i16,
}

struct NnuePolicyNet {
  from_weights: [VecI8; 64],
  from_bias:    [i16; 64],
  to_weights:   [VecI8; 64],
  to_bias:      [i16; 64],
}

pub struct Nnue<'a> {
  layers:       &'a [[VecI16; LINEAR_STATE_VECTOR_COUNT]],
  value_nets:   [NnueValueNet; 32],
  policy_nets:  [NnuePolicyNet; 32],
  main_bias:    [VecI16; LINEAR_STATE_VECTOR_COUNT],
  linear_state: [VecI16; LINEAR_STATE_VECTOR_COUNT],
  layer_count:  i32,
}

impl<'a> Nnue<'a> {
  pub fn new(state: &State, file_contents: &'a [u8]) -> Self {
    //let linear_state = get_bias1d_i16!(PARAMS_MAIN_EMBED_BIAS, LINEAR_STATE_SIZE);
    let weights = NnueWeights::from_file(file_contents);
    let layers = weights.get_veci16_2d::<LINEAR_STATE_VECTOR_COUNT>(
      "main_embed.weight",
      &[106496, LINEAR_STATE_SIZE],
      8,
    );
    let main_bias = weights.get_veci16("main_bias", &[LINEAR_STATE_SIZE], 8);
    let mut value_nets = vec![];
    let mut policy_nets = vec![];
    for i in 0..32 {
      value_nets.push(NnueValueNet {
        l0_weights: weights
          .get_veci8_2d::<16>(&format!("value{i}.0.w"), &[16, 256], 7)
          .try_into()
          .unwrap(),
        l0_bias:    weights.get_i16(&format!("value{i}.0.b"), &[16], 14).try_into().unwrap(),
        l1_weights: weights.get_veci8(&format!("value{i}.1.w"), &[32, 16], 7).try_into().unwrap(),
        l1_bias:    weights.get_i16(&format!("value{i}.1.b"), &[32], 14).try_into().unwrap(),
        l2_weights: weights.get_veci8(&format!("value{i}.2.w"), &[1, 32], 7).try_into().unwrap(),
        l2_bias:    weights.get_i16(&format!("value{i}.2.b"), &[1], 14)[0],
      });
      policy_nets.push(NnuePolicyNet {
        from_weights: weights
          .get_veci8(&format!("policy_from{i}.0.w"), &[64, 16], 6)
          .try_into()
          .unwrap(),
        from_bias:    weights
          .get_i16(&format!("policy_from{i}.0.b"), &[64], 13)
          .try_into()
          .unwrap(),
        to_weights:   weights
          .get_veci8(&format!("policy_to{i}.0.w"), &[64, 16], 6)
          .try_into()
          .unwrap(),
        to_bias:      weights.get_i16(&format!("policy_to{i}.0.b"), &[64], 13).try_into().unwrap(),
      });
    }

    let mut this = Self {
      layers,
      value_nets: value_nets.try_into().map_err(|_| unreachable!()).unwrap(),
      policy_nets: policy_nets.try_into().map_err(|_| unreachable!()).unwrap(),
      main_bias: main_bias.try_into().unwrap(),
      linear_state: [veci16_zeros(); LINEAR_STATE_VECTOR_COUNT],
      layer_count: 0,
    };
    this.recompute_linear_state(state);
    this
  }

  pub fn dump_state(&self) {
    // Cast our linear state to a slice of i16s.
    let linear_state = unsafe {
      std::slice::from_raw_parts(self.linear_state.as_ptr() as *const i16, LINEAR_STATE_SIZE)
    };
    print!("[");
    for i in 0..LINEAR_STATE_SIZE {
      print!("{}, ", linear_state[i]);
    }
    println!("]");
  }

  pub fn recompute_linear_state(&mut self, state: &State) {
    self.linear_state = self.main_bias;
    self.layer_count = 0;
    compute_state_layers(state, |layer| self.apply_layer::<false, false>(layer)).unwrap_or(())
  }

  pub fn compute_from_indices(&mut self, indices: &[i32]) {
    self.linear_state = self.main_bias;
    self.layer_count = 0;
    for &index in indices {
      if index != -1 {
        self.apply_layer::<false, false>(index as LayerIndex);
      }
    }
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

  pub fn apply_adjustment<const UNDO: bool>(&mut self, state: &State, adjustment: &NnueAdjustment) {
    let king_positions = get_king_positions(state);
    match adjustment {
      NnueAdjustment::DuckCreation { to } => {
        let (black_layer, white_layer) = to.get_layers(king_positions);
        self.apply_layer::<UNDO, false>(black_layer);
        self.apply_layer::<UNDO, false>(white_layer);
      }
      NnueAdjustment::Normal { from, to, capture } => {
        let (black_layer, white_layer) = from.get_layers(king_positions);
        self.apply_layer::<UNDO, true>(black_layer);
        self.apply_layer::<UNDO, true>(white_layer);
        let (black_layer, white_layer) = to.get_layers(king_positions);
        self.apply_layer::<UNDO, false>(black_layer);
        self.apply_layer::<UNDO, false>(white_layer);
        if let Some(capture) = capture {
          let (black_layer, white_layer) = capture.get_layers(king_positions);
          self.apply_layer::<UNDO, true>(black_layer);
          self.apply_layer::<UNDO, true>(white_layer);
        }
      }
      NnueAdjustment::KingInvolved => {
        //panic!("King involved adjustment not implemented");
        self.recompute_linear_state(state);
      }
    }
  }

  #[inline]
  fn apply_layer<const UNDO: bool, const SUB: bool>(&mut self, layer: LayerIndex) {
    if UNDO != SUB {
      self.layer_count -= 1;
    } else {
      self.layer_count += 1;
    }
    for i in 0..LINEAR_STATE_VECTOR_COUNT {
      if UNDO != SUB {
        self.linear_state[i] = veci16_sub(self.linear_state[i], self.layers[layer as usize][i]);
      } else {
        self.linear_state[i] = veci16_add(self.linear_state[i], self.layers[layer as usize][i]);
      }
    }
  }

  pub fn evaluate_policy(&self, state: &State, square_from: &mut [i16], square_to: &mut [i16]) {
    let network_index = get_net_index(state, self.layer_count);
    let policy_net = &self.policy_nets[network_index];

    //// Cast self.linear_state to &[i16].
    //let linear_state = unsafe {
    //  std::slice::from_raw_parts(self.linear_state.as_ptr() as *const i16, LINEAR_STATE_SIZE)
    //};
    //print!("linear_state[16:48] = [");
    //for i in 16..48 {
    //  print!("{}, ", linear_state[i] as f32 / (1 << 8) as f32);
    //}
    //println!("]");

    let from_data_simd: VecI8 = {
      let v0: VecI16 = veci16_shr_bias_to_main(self.linear_state[2]);
      let v1: VecI16 = veci16_shr_bias_to_main(self.linear_state[3]);
      vec_narrow_pair(v0, v1)
    };
    let to_data_simd: VecI8 = {
      let v0: VecI16 = veci16_shr_bias_to_main(self.linear_state[4]);
      let v1: VecI16 = veci16_shr_bias_to_main(self.linear_state[5]);
      vec_narrow_pair(v0, v1)
    };
    for i in 0..64 {
      square_from[i] = veci16_horizontal_sat_sum(vec_matmul_sat_fma(
        veci16_zeros(),
        from_data_simd,
        policy_net.from_weights[i],
      ))
      .saturating_add(policy_net.from_bias[i]);
      square_to[i] = veci16_horizontal_sat_sum(vec_matmul_sat_fma(
        veci16_zeros(),
        to_data_simd,
        policy_net.to_weights[i],
      ))
      .saturating_add(policy_net.to_bias[i]);
    }
  }

  pub fn evaluate_value(&self, state: &State) -> IntEvaluation {
    if let Some(terminal_eval) = eval_terminal_state(state) {
      return terminal_eval;
    }

    // Compute the game phase.
    //let game_phase = state.get_occupied().count_ones() / 8;
    // FIXME: For now just always use phase 3.
    //let game_phase = 3;
    //let network_index = state.turn as usize + 2 * state.is_duck_move as usize + 4 * game_phase;
    // FIXME: I need to reimplement everything here.
    let network_index = get_net_index(state, self.layer_count);
    let value_net = &self.value_nets[network_index];

    // Get the first i16 from the first vector of the linear state by unsafe casting.
    let linear_state_i16 = unsafe {
      std::slice::from_raw_parts(self.linear_state.as_ptr() as *const i16, LINEAR_STATE_SIZE)
    };
    let psqt = linear_state_i16[0] as i32;

    // The horizontal sum of all of the entries in accum[i] represents the output of layer 0.
    let mut accums0 = [veci16_zeros(); 16];
    for i in 0..LINEAR_STATE_VECTOR_COUNT / 2 {
      let v0: VecI16 = veci16_shr_bias_to_main(self.linear_state[2 * i + 0]);
      let v1: VecI16 = veci16_shr_bias_to_main(self.linear_state[2 * i + 1]);
      let v = vec_narrow_pair(v0, v1);
      // Print out this block:
      //let transmuted = unsafe { std::mem::transmute::<VecI8, [i8; 16]>(v) };
      //println!("Block {} = {:?}", i, transmuted);
      for j in 0..16 {
        let block = value_net.l0_weights[j][i];
        accums0[j] = vec_matmul_sat_fma(accums0[j], v, block);
      }
    }
    // We now horizontally sum the accums0, add the bias, and apply the activation function.
    let mut layer0: [i8; 16] = [0; 16];
    for i in 0..16 {
      let intermediate = veci16_horizontal_sat_sum(accums0[i]).saturating_add(value_net.l0_bias[i]);
      //println!("intermediate: {} (added {})", intermediate, value_net.l0_bias[i]);
      layer0[i] = (intermediate >> 7).clamp(0, 127) as i8;
    }
    let layer0_simd: VecI8 = unsafe { std::mem::transmute(layer0) };
    // Print out this intermediate.
    //let layer0_i8 = unsafe { std::slice::from_raw_parts(&layer0_simd as *const VecI8 as *const i8, 16) };
    //print!("layer0 out = [");
    //for i in 0..16 {
    //  print!("{}, ", layer0_i8[i]);
    //}
    //println!("]");

    // Second layer
    let mut accums1 = [veci16_zeros(); 32];
    for j in 0..32 {
      let block = value_net.l1_weights[j];
      accums1[j] = vec_matmul_sat_fma(accums1[j], layer0_simd, block);
    }
    let mut layer1: [i8; 32] = [0; 32];
    for i in 0..32 {
      let intermediate = veci16_horizontal_sat_sum(accums1[i]).saturating_add(value_net.l1_bias[i]);
      layer1[i] = (intermediate >> 7).clamp(0, 127) as i8;
    }
    let layer1_simd: [VecI8; 2] = unsafe { std::mem::transmute(layer1) };
    // Print out this intermediate.
    //let layer1_i8 = unsafe { std::slice::from_raw_parts(&layer1_simd as *const [VecI8; 2] as *const i8, 32) };
    //print!("layer1 out = [");
    //for i in 0..32 {
    //  print!("{}, ", layer1_i8[i]);
    //}
    //println!("]");

    // Final layer
    // Print out all of the inputs and weights.
    //let l2_weights_i8 = unsafe { std::slice::from_raw_parts(&value_net.l2_weights as *const VecI8 as *const i8, 32) };
    //print!("l2_weights = [");
    //for i in 0..32 {
    //  print!("{}, ", l2_weights_i8[i]);
    //}
    //println!("]");
    let mut final_accum = veci16_zeros();
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[0], value_net.l2_weights[0]);
    final_accum = vec_matmul_sat_fma(final_accum, layer1_simd[1], value_net.l2_weights[1]);
    let final_accum = veci16_horizontal_sat_sum(final_accum) as i32;
    //println!("final_accum: {} (wo/ bias)", final_accum);
    let final_accum = final_accum + value_net.l2_bias as i32;
    //println!("final_accum: {} (w/ bias = {})", final_accum, net.l2_bias);
    //println!("psqt: {}", psqt);
    let final_accum = final_accum + (psqt << 6);
    //println!("final_accum: {} (w/ psqt = {})", final_accum, psqt << 3);

    // We now negate the value if it's black's turn.
    match state.turn {
      Player::White => final_accum,
      Player::Black => -final_accum,
    }

    //let network_output = (final_accum as f32 / (1 << 14) as f32).tanh();
    //self.value = (self.outputs[64].tanh() * 1000.0) as i32;

    //IntEvaluation {
    //  expected_score: (network_output + 1.0) / 2.0,
    //  perspective_player: Player::White
    //  //perspective_player: state.turn,
    //}
  }
}

#[cfg(test)]
mod tests {
  #[test]
  fn test_veci16_horizontal_sat_sum() {
    use super::*;
    let data: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let v = unsafe { std::mem::transmute::<[i16; 8], VecI16>(data) };
    let sum = veci16_horizontal_sat_sum(v);
    assert_eq!(sum, 36);
  }
}

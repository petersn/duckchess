use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize, ser::SerializeSeq};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[derive(Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CastlingRights {
    king_side:  bool,
    queen_side: bool,
}

#[derive(Hash)]
struct BitBoard(u64);

impl Serialize for BitBoard {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut bytes = serializer.serialize_seq(Some(8))?;
        for i in 0..8 {
            let byte = (self.0 >> (i * 8)) as u8;
            bytes.serialize_element(&byte)?;
        }
        bytes.end()
    }
}

impl<'de> Deserialize<'de> for BitBoard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let mut board = 0;
        for (i, byte) in bytes.iter().enumerate() {
            board |= (*byte as u64) << (i * 8);
        }
        Ok(BitBoard(board))
    }
}


#[derive(Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct State {
    pawns:           [BitBoard; 2],
    knights:         [BitBoard; 2],
    bishops:         [BitBoard; 2],
    rooks:           [BitBoard; 2],
    queens:          [BitBoard; 2],
    kings:           [BitBoard; 2],
    duck:            BitBoard,
    en_passant:      BitBoard,
    castling_rights: [CastlingRights; 2],
    is_duck_move:    bool,
}

fn starting_state() -> State {
    State {
        pawns:           [BitBoard(0x000000000000ff00), BitBoard(0x00ff000000000000)],
        knights:         [BitBoard(0x0000000000000042), BitBoard(0x4200000000000000)],
        bishops:         [BitBoard(0x0000000000000024), BitBoard(0x2400000000000000)],
        rooks:           [BitBoard(0x0000000000000081), BitBoard(0x8100000000000000)],
        queens:          [BitBoard(0x0000000000000008), BitBoard(0x0800000000000000)],
        kings:           [BitBoard(0x0000000000000010), BitBoard(0x1000000000000000)],
        duck:            BitBoard(0),
        en_passant:      BitBoard(0),
        castling_rights: [
            CastlingRights {
                king_side:  true,
                queen_side: true,
            },
            CastlingRights {
                king_side:  true,
                queen_side: true,
            },
        ],
        is_duck_move:    false,
    }
}

#[wasm_bindgen]
pub struct Engine {
    state: State,
}

#[wasm_bindgen]
impl Engine {
    pub fn get_state(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.state).unwrap_or_else(|e| {
            log(&format!("Failed to serialize state: {}", e));
            JsValue::NULL
        })
    }

    pub fn set_state(&mut self, state: JsValue) {
        self.state = serde_wasm_bindgen::from_value(state).expect("serialization");
    }
}

#[wasm_bindgen]
pub fn new_engine() -> Engine {
    Engine {
        state: starting_state(),
    }
}

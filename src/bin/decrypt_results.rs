use serde_json::{self, from_str};
use std::{env, fs};
use tfhe::prelude::*;
use tfhe::safe_serialization::safe_deserialize;
use tfhe::{ClientKey, FheInt16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client_key: ClientKey = safe_deserialize(
        fs::read("./keys/client_key.bin").unwrap().as_slice(),
        1 << 30,
    )
    .unwrap();

    for i in 0..9 {
        let input_buffer = fs::read(format!("./keys/layer_2_{}.bin", i)).unwrap();
        let encrypted_input: FheInt16 = bincode::deserialize_from(&mut input_buffer.as_slice())?;
        let result: i16 = encrypted_input.decrypt(&client_key);

        println!("{} {}", i, result);
    }

    Ok(())
}

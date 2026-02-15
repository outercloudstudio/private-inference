use serde_json::{self, from_str};
use std::{env, fs};
use tfhe::prelude::*;
use tfhe::safe_serialization::safe_deserialize;
use tfhe::{ClientKey, FheInt16};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let clear_inputs: Vec<i16> = from_str(&args[1]).expect("Failed to parse JSON");

    let client_key: ClientKey = safe_deserialize(
        fs::read("./keys/client_key.bin").unwrap().as_slice(),
        1 << 30,
    )
    .unwrap();

    println!("{:?}", clear_inputs);

    let mut serialized_encrypted_inputs = Vec::new();

    bincode::serialize_into(
        &mut serialized_encrypted_inputs,
        &(clear_inputs.len() as i16),
    )?;

    for (_, &value) in clear_inputs.iter().enumerate() {
        let encrypted = FheInt16::try_encrypt(value, &client_key)?;

        bincode::serialize_into(&mut serialized_encrypted_inputs, &encrypted)?;
    }

    fs::write(
        format!("./keys/encrypted_inputs.bin"),
        &serialized_encrypted_inputs,
    )?;

    println!("Image encrypted!");

    Ok(())
}

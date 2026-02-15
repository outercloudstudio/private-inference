use base64::Engine;
use base64::engine::general_purpose;
use serde::{Deserialize, Serialize};
use serde_json::{self, from_str};
use std::{env, fs};
use tfhe::boolean::prelude::{BinaryBooleanGates, ServerKey};
use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheBool, FheInt8, FheInt16, generate_keys, set_server_key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let clear_inputs: Vec<i16> = from_str(&args[1]).expect("Failed to parse JSON");

    let config = ConfigBuilder::default().build();

    let (client_key, server_key) = generate_keys(config);

    let mut encrypted_inputs: Vec<FheInt16> = Vec::new();

    println!("{:?}", clear_inputs);

    for (i, &value) in clear_inputs.iter().enumerate() {
        let encrypted = FheInt16::try_encrypt(value, &client_key)?;
        encrypted_inputs.push(encrypted);

        println!("Encrypted {}/{}", i + 1, clear_inputs.len());
    }

    let serialized_inputs = serde_json::to_vec(&encrypted_inputs)?;
    fs::write("./keys/encrypted_inputs.bin", &serialized_inputs)?;

    let serialized_server_key = serde_json::to_vec(&server_key)?;
    fs::write("./keys/server_key.bin", &serialized_server_key)?;

    let serialized_client_key = serde_json::to_vec(&client_key)?;
    fs::write("./keys/client_key.bin", &serialized_client_key)?;

    println!("Keys saved!");

    Ok(())
}

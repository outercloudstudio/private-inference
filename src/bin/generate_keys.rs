use serde_json::{self, from_str};
use std::{env, fs};
use tfhe::prelude::*;
use tfhe::safe_serialization::safe_serialize;
use tfhe::{ConfigBuilder, FheInt16, generate_keys};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let clear_inputs: Vec<i16> = from_str(&args[1]).expect("Failed to parse JSON");

    let config = ConfigBuilder::default().build();

    let (client_key, server_key) = generate_keys(config);

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

    let encrypted_zero = FheInt16::try_encrypt(0i8, &client_key)?;
    let mut serialized_encrypted_zero = Vec::new();
    bincode::serialize_into(&mut serialized_encrypted_zero, &encrypted_zero)?;
    fs::write(
        format!("./keys/encrypted_zero.bin"),
        &serialized_encrypted_zero,
    )?;

    let mut server_key_buffer = vec![];
    safe_serialize(&server_key, &mut server_key_buffer, 1 << 30).unwrap();
    fs::write("./keys/server_key.bin", &server_key_buffer)?;

    let mut client_key_buffer = vec![];
    safe_serialize(&client_key, &mut client_key_buffer, 1 << 30).unwrap();
    fs::write("./keys/client_key.bin", &client_key_buffer)?;

    println!("Keys saved!");

    Ok(())
}

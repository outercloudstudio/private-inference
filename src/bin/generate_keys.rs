use serde_json::{self, from_str};
use std::{env, fs};
use tfhe::prelude::*;
use tfhe::safe_serialization::safe_serialize;
use tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::{ConfigBuilder, FheInt16, generate_keys};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConfigBuilder::default().build();

    let (client_key, server_key) = generate_keys(config);

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

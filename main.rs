use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheBool, FheInt8, FheUint8, generate_keys, set_server_key};

fn relu(value: FheInt8, encrypted_zero: &FheInt8) -> FheInt8 {
    let comparison = value.ge(encrypted_zero);

    return comparison.select(&value, encrypted_zero);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Basic configuration to use homomorphic integers
    let config = ConfigBuilder::default().build();

    // Key generation
    let (client_key, server_keys) = generate_keys(config);

    let clear_a = 1i8;
    let clear_b = -1i8;
    let clear_zero = 0i8;

    let encrypted_a = FheInt8::try_encrypt(clear_a, &client_key)?;
    let encrypted_b = FheInt8::try_encrypt(clear_b, &client_key)?;
    let encrypted_zero = FheInt8::try_encrypt(clear_zero, &client_key)?;

    // On the server side:
    set_server_key(server_keys);

    let encrypted_res_mul = &encrypted_a * &encrypted_b;

    let clear_res: i8 = encrypted_res_mul.decrypt(&client_key);
    assert_eq!(clear_res, -1_i8);

    println!("{}", clear_res);

    Ok(())
}

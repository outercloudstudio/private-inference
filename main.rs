use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheBool, FheInt8, FheUint8, generate_keys, set_server_key};

use serde::{Deserialize, Serialize};
use serde_json::{self, from_str};

#[derive(Debug, Deserialize, Serialize)]
struct BinaryLayer {
    weight: Vec<Vec<i32>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LinearLayer {
    weight: Vec<Vec<i32>>,
    bias: Vec<i32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Model {
    fc1: BinaryLayer,
    fc2: BinaryLayer,
    fc3: BinaryLayer,
    fc4: LinearLayer,
}

fn binary_node(inputs: &Vec<FheInt8>, weights: &Vec<FheInt8>) -> FheInt8 {
    let mut accumulator = &inputs[0] * &weights[0];

    for i in 1..inputs.len() {
        let product = &inputs[i] * &weights[i];

        accumulator = accumulator + product;
    }

    return accumulator;
}

fn relu(value: FheInt8, encrypted_zero: &FheInt8) -> FheInt8 {
    let comparison = value.ge(encrypted_zero);

    return comparison.select(&value, encrypted_zero);
}

const JSON_STR: &str = include_str!("binary_model.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model: Model = from_str(JSON_STR).expect("Failed to parse JSON");

    let config = ConfigBuilder::default().build();

    let (client_key, server_keys) = generate_keys(config);

    let clear_a = 1i8;
    let clear_b = -1i8;
    let clear_zero = 0i8;

    let encrypted_a = FheInt8::try_encrypt(clear_a, &client_key)?;
    let encrypted_b = FheInt8::try_encrypt(clear_b, &client_key)?;
    let encrypted_zero = FheInt8::try_encrypt(clear_zero, &client_key)?;

    // On the server side:
    set_server_key(server_keys);

    let mut inputs: Vec<FheInt8> = Vec::new();

    for i in 0..196 {
        let encrypted_input = FheInt8::try_encrypt(0i8, &client_key)?;

        inputs.push(encrypted_input);
    }

    let mut weights: Vec<FheInt8> = Vec::new();

    for i in 0..196 {
        let encrypted_weight = FheInt8::try_encrypt(model.fc1.weight[0][i], &client_key)?;

        weights.push(encrypted_weight);
    }

    // let encrypted_res_mul = &encrypted_a * &encrypted_b;

    // let clear_res: i8 = encrypted_res_mul.decrypt(&client_key);
    // assert_eq!(clear_res, -1_i8);

    // println!("{}", clear_res);

    Ok(())
}

use tfhe::boolean::prelude::{BinaryBooleanGates, ServerKey};
use tfhe::prelude::*;
use tfhe::{ConfigBuilder, FheBool, FheInt8, FheInt16, generate_keys, set_server_key};

use serde::{Deserialize, Serialize};
use serde_json::{self, from_str};

#[derive(Debug, Deserialize, Serialize)]
struct BinaryLayer {
    weight: Vec<Vec<i16>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LinearLayer {
    weight: Vec<Vec<i16>>,
    bias: Vec<i16>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Model {
    fc1: BinaryLayer,
    fc2: BinaryLayer,
    // fc3: BinaryLayer,
    // fc4: LinearLayer,
    fc3: LinearLayer,
}

fn binary_node(inputs: &Vec<FheInt16>, weights: &Vec<i16>) -> FheInt16 {
    let mut sum = &inputs[0] * weights[0];

    for i in 1..inputs.len() {
        sum = sum + &inputs[i] * weights[i];
    }

    return sum;
}

fn binary_node_clear(inputs: &Vec<i16>, weights: &Vec<i16>) -> i16 {
    let mut sum = inputs[0] * weights[0];

    for i in 1..inputs.len() {
        sum = sum + inputs[i] * weights[i];
    }

    return sum;
}

fn relu(value: FheInt16, encrypted_zero: &FheInt16) -> FheInt16 {
    let comparison = value.ge(encrypted_zero);

    return comparison.select(&value, encrypted_zero);
}

const JSON_STR: &str = include_str!("binary_model.json");

fn run_clear_model() {
    let model: Model = from_str(JSON_STR).expect("Failed to parse JSON");

    let clear_inputs: Vec<i16> = vec![
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
        -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,
    ];

    println!("{:?}", clear_inputs);

    let mut clear_layer_0: Vec<i16> = Vec::new();

    for i in 0..64 {
        let mut weights: Vec<i16> = Vec::new();

        for j in 0..49 {
            weights.push(model.fc1.weight[i][j] as i16);
        }

        clear_layer_0.push(i16::max(0, binary_node_clear(&clear_inputs, &weights)));
    }

    println!("{:?}", clear_layer_0);

    let mut clear_layer_1: Vec<i16> = Vec::new();

    for i in 0..64 {
        let mut weights: Vec<i16> = Vec::new();

        for j in 0..64 {
            weights.push(model.fc2.weight[i][j] as i16);
        }

        clear_layer_1.push(i16::max(0, binary_node_clear(&clear_layer_0, &weights)));
    }

    println!("{:?}", clear_layer_1);

    let mut clear_layer_2: Vec<i16> = Vec::new();

    for i in 0..10 {
        let mut sum: i16 = 0i16;

        for j in 0..64 {
            sum += model.fc3.weight[i][j] as i16 * clear_layer_1[j] as i16;
        }

        clear_layer_2.push(sum + model.fc3.bias[i]);
    }

    println!("{:?}", clear_layer_2);

    let mut max_index = 0;

    for i in 1..10 {
        if clear_layer_2[i] > clear_layer_2[max_index] {
            max_index = i;
        }
    }

    println!("{}", max_index);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_clear_model();

    let model: Model = from_str(JSON_STR).expect("Failed to parse JSON");

    let config = ConfigBuilder::default().build();

    let (client_key, server_keys) = generate_keys(config);

    let encrypted_zero = FheInt16::try_encrypt(0i8, &client_key)?;

    // On the server side:
    set_server_key(server_keys);

    let clear_inputs: Vec<i16> = vec![
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
        -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,
    ];

    let mut inputs: Vec<FheInt16> = Vec::new();

    for i in 0..49 {
        let encrypted_input = FheInt16::try_encrypt(clear_inputs[i], &client_key)?;

        inputs.push(encrypted_input);
    }

    let mut layer_0: Vec<FheInt16> = Vec::new();

    for i in 0..64 {
        let mut weights: Vec<i16> = Vec::new();

        for j in 0..49 {
            weights.push(model.fc1.weight[i][j]);
        }

        let result = relu(binary_node(&inputs, &weights), &encrypted_zero);

        let clear_result: i16 = result.decrypt(&client_key);

        println!("{}", clear_result);

        layer_0.push(result);
    }

    // let clear_result: i8 = result.decrypt(&client_key);

    // println!("{}", clear_result);

    Ok(())
}

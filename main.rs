use tfhe::boolean::prelude::{BinaryBooleanGates, ServerKey};
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

fn binary_node(
    inputs: &Vec<FheBool>,
    weights: &Vec<bool>,
    encrypted_one: &FheInt8,
    encrypted_zero: &FheInt8,
    encrypted_neg: &FheInt8,
) -> FheInt8 {
    let mut accumulator = encrypted_zero.clone();

    for i in 0..inputs.len() {
        let xnor = if weights[i] {
            inputs[i].clone()
        } else {
            !&inputs[i]
        };

        let contribution = xnor.select(&encrypted_one.clone(), &encrypted_neg.clone());

        accumulator = accumulator + contribution;
    }

    return accumulator;
}

fn binary_node_clear(inputs: &Vec<i16>, weights: &Vec<i16>) -> i16 {
    let mut sum = inputs[0] * weights[0];

    for i in 1..inputs.len() {
        sum = sum + inputs[i] * weights[i];
    }

    return sum;

    // let mut accumulator = 0;

    // for i in 0..inputs.len() {
    //     let xnor = if weights[i] { inputs[i] } else { !&inputs[i] };

    //     let contribution = if xnor { 1 } else { -1 };

    //     accumulator = accumulator + contribution;
    // }

    // return accumulator;
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

    let encrypted_zero = FheInt8::try_encrypt(0i8, &client_key)?;
    let encrypted_one = FheInt8::try_encrypt(1i8, &client_key)?;
    let encrypted_neg = FheInt8::try_encrypt(-1i8, &client_key)?;

    // On the server side:
    set_server_key(server_keys);

    let mut inputs: Vec<FheBool> = Vec::new();
    let mut clear_inputs: Vec<i16> = vec![
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1,
        -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1,
    ];

    for i in 0..49 {
        let encrypted_input = FheBool::try_encrypt(true, &client_key)?;

        inputs.push(encrypted_input);
    }

    // let mut weights: Vec<bool> = Vec::new();

    // for i in 0..49 {
    //     weights.push(model.fc1.weight[0][i] == 1);
    // }

    // println!("binary_node!");

    // let result = binary_node(
    //     &inputs,
    //     &weights,
    //     &encrypted_one,
    //     &encrypted_zero,
    //     &encrypted_neg,
    // );

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

    // let clear_res: i8 = result.decrypt(&client_key);

    // println!("{:?}", weights);
    // println!("{:?}", clear_inputs);
    // println!("{} {}", clear_res, check);

    Ok(())
}

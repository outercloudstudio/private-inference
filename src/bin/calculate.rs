use serde::{Deserialize, Serialize};
use serde_json::{self, from_str};
use std::env;
use std::fs;
use std::io::Cursor;
use tfhe::prelude::*;
use tfhe::safe_serialization::safe_deserialize_conformant;
use tfhe::shortint::prelude::PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::{ConfigBuilder, FheInt16, ServerKey, set_server_key};

#[derive(Deserialize, Serialize)]
struct Location {
    node: i16,
    layer: i16,
}

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
    fc3: LinearLayer,
}

fn binary_node(inputs: &Vec<FheInt16>, weights: &Vec<i16>) -> FheInt16 {
    let mut sum = &inputs[0] * weights[0];

    for i in 1..inputs.len() {
        sum = sum + &inputs[i] * weights[i];
    }

    return sum;
}

fn linear_node(inputs: &Vec<FheInt16>, weights: &Vec<i16>, bias: i16) -> FheInt16 {
    let mut sum = &inputs[0] * weights[0];

    for i in 1..inputs.len() {
        sum = sum + &inputs[i] * weights[i];
    }

    return sum + bias;
}

fn relu(value: FheInt16, encrypted_zero: &FheInt16) -> FheInt16 {
    return value.max(encrypted_zero);
}

const JSON_STR: &str = include_str!("../../binary_model.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let location: Location = from_str(&args[1]).expect("Failed to parse JSON");

    let config = ConfigBuilder::default().build();

    let server_key: ServerKey = safe_deserialize_conformant(
        fs::read("./keys/server_key.bin").unwrap().as_slice(),
        1 << 30,
        &config.into(),
    )
    .unwrap();

    let encrypted_zero: FheInt16 =
        bincode::deserialize_from(fs::read("./keys/encrypted_zero.bin").unwrap().as_slice())
            .unwrap();

    let mut inputs: Vec<FheInt16> = Vec::new();

    if location.layer == 0 {
        let inputs_buffer = fs::read("./keys/encrypted_inputs.bin").unwrap();
        let mut inputs_cursor = Cursor::new(inputs_buffer.as_slice());
        let inputs_count: i16 = bincode::deserialize_from(&mut inputs_cursor).unwrap();

        for _ in 0..inputs_count {
            let encrypted_input: FheInt16 = bincode::deserialize_from(&mut inputs_cursor)?;

            inputs.push(encrypted_input);
        }
    } else if location.layer == 1 {
        for i in 0..32 {
            let input_buffer = fs::read(format!("./keys/layer_0_{}.bin", i)).unwrap();
            let encrypted_input: FheInt16 =
                bincode::deserialize_from(&mut input_buffer.as_slice())?;

            inputs.push(encrypted_input);
        }
    } else if location.layer == 2 {
        for i in 0..32 {
            let input_buffer = fs::read(format!("./keys/layer_1_{}.bin", i)).unwrap();
            let encrypted_input: FheInt16 =
                bincode::deserialize_from(&mut input_buffer.as_slice())?;

            inputs.push(encrypted_input);
        }
    }

    set_server_key(server_key);

    let model: Model = from_str(JSON_STR).expect("Failed to parse JSON");

    if location.layer == 0 {
        println!("Executing Layer 0!");

        let mut weights: Vec<i16> = Vec::new();

        for j in 0..49 {
            weights.push(model.fc1.weight[location.node as usize][j]);
        }

        let result = relu(binary_node(&inputs, &weights), &encrypted_zero);

        println!("Got result!");

        let mut serialized_result = Vec::new();
        bincode::serialize_into(&mut serialized_result, &result)?;
        fs::write(
            format!("./keys/layer_{}_{}.bin", location.layer, location.node),
            &serialized_result,
        )?;
    } else if location.layer == 1 {
        println!("Executing Layer 1!");

        let mut weights: Vec<i16> = Vec::new();

        for j in 0..32 {
            weights.push(model.fc2.weight[location.node as usize][j]);
        }

        let result = relu(binary_node(&inputs, &weights), &encrypted_zero);

        println!("Got result!");

        let mut serialized_result = Vec::new();
        bincode::serialize_into(&mut serialized_result, &result)?;
        fs::write(
            format!("./keys/layer_{}_{}.bin", location.layer, location.node),
            &serialized_result,
        )?;
    } else if location.layer == 2 {
        println!("Executing Layer 2!");

        let mut weights: Vec<i16> = Vec::new();

        for j in 0..32 {
            weights.push(model.fc3.weight[location.node as usize][j]);
        }

        let result = linear_node(&inputs, &weights, model.fc3.bias[location.node as usize]);

        println!("Got result!");

        let mut serialized_result = Vec::new();
        bincode::serialize_into(&mut serialized_result, &result)?;
        fs::write(
            format!("./keys/layer_{}_{}.bin", location.layer, location.node),
            &serialized_result,
        )?;
    }

    Ok(())
}

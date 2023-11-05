mod matrix;
mod nn;

use nn::ModelParams;
use serde::Deserialize;

#[derive(Deserialize)]
struct Sample {
    input: Vec<f32>,
    output: Vec<f32>,
}

fn samples_to_training_data(samples: &[Sample]) -> (Vec<&[f32]>, Vec<&[f32]>) {
    let mut inputs = Vec::with_capacity(samples.len());
    let mut outputs = Vec::with_capacity(samples.len());

    for sample in samples {
        inputs.push(&sample.input[..]);
        outputs.push(&sample.output[..]);
    }

    (inputs, outputs)
}

fn main() {
    let nn_structure = [2, 2, 1];
    let mut m = ModelParams::new(&nn_structure);
    m.randomize(0.0, 1.0);

    let samples: Vec<Sample> = serde_json::from_str(include_str!("samples.json")).unwrap();
    let (train_in, train_out) = samples_to_training_data(&samples);

    const EPS: f32 = 0.1;
    const RATE: f32 = 1.0;

    for i in 0..10000 {
        println!("{}: cost = {}", i, nn::cost(&m, &train_in, &train_out));
        let grad = nn::finite_diff(&m, &train_in, &train_out, EPS);
        m = nn::descend(&m, &grad, RATE);
    }

    println!();
    for i in 0..2 {
        for j in 0..2 {
            println!(
                "{} ^ {} = {:?}",
                i,
                j,
                nn::forward(&m, &[i as f32, j as f32])
            );
        }
    }

    println!("\n{}", m);
}

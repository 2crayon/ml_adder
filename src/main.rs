mod matrix;
mod nn;

use nn::ModelParams;

static SEP: &str = "----------------------------------------";

fn main() {
    let nn_structure = [2, 1];
    let mut m = ModelParams::new(&nn_structure);
    m.randomize(0.0, 1.0);

    let train_in = &[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        .iter()
        .map(|arr| arr.as_slice())
        .collect::<Vec<&[f32]>>()[..];
    let train_out = &[[0.0], [1.0], [1.0], [1.0]]
        .iter()
        .map(|arr| arr.as_slice())
        .collect::<Vec<&[f32]>>()[..];

    const EPS: f32 = 0.1;
    const RATE: f32 = 1.0;

    for i in 0..10_000 {
        println!("{}: cost = {}", i, nn::cost(&m, train_in, train_out));
        let grad = nn::finite_diff(&m, train_in, train_out, EPS);
        m = nn::descend(&m, &grad, RATE);
    }

    println!("{}", SEP);
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

    println!("{SEP}\n{}", m);
}

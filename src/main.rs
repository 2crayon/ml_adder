mod matrix;
mod nn;

use matrix::Matrix;
use nn::ModelParams;

static SEP: &str = "----------------------------------------";

fn main() {
    let mut m = ModelParams::new();
    m.randomize(0.0, 1.0);

    let t_in = Matrix::from_slice(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);
    let t_out = Matrix::from_slice(&[&[0.0], &[1.0], &[1.0], &[0.0]]);

    println!("cost = {}", nn::cost(&m, t_in, t_out));
    println!("{}", SEP);

    for i in 0..2 {
        for j in 0..2 {
            let mut a0 = Matrix::new(1, 2);
            a0[0][0] = i as f32;
            a0[0][1] = j as f32;

            println!("{} ^ {} = {}", i, j, nn::forward(&m, &a0));
        }
    }
}

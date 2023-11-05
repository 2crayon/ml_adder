use std::fmt;

use crate::matrix::Matrix;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Clone, Debug)]
pub struct ModelParams {
    w: Vec<Matrix>,
    b: Vec<Matrix>,
}

impl ModelParams {
    pub fn new(structure: &[usize]) -> Self {
        assert!(structure.len() > 0);

        let layer_count = structure.len() - 1;
        let mut w = Vec::with_capacity(layer_count);
        let mut b = Vec::with_capacity(layer_count);

        for i in 0..layer_count {
            w.push(Matrix::new(structure[i], structure[i + 1]));
            b.push(Matrix::new(1, structure[i + 1]));
        }

        ModelParams { w, b }
    }

    pub fn randomize(&mut self, min: f32, max: f32) {
        let layer_count = self.w.len();

        for i in 0..layer_count {
            self.w[i].randomize(min, max);
            self.b[i].randomize(min, max);
        }
    }

    pub fn structure(&self) -> Vec<usize> {
        // i don't get this

        let mut result = Vec::with_capacity(self.w.len() + 1);
        result.push(self.w[0].row_count());
        for i in 0..self.w.len() {
            result.push(self.w[i].col_count());
        }
        result
    }
}

impl fmt::Display for ModelParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layers_count = self.w.len();

        for i in 0..layers_count {
            write!(f, "w[{}]\n{}", i, self.w[i])?;
            write!(f, "b[{}]\n{}", i, self.b[i])?;
        }

        Ok(())
    }
}

pub fn forward(m: &ModelParams, inputs: &[f32]) -> Vec<f32> {
    let layers_count = m.w.len();

    let mut a = Matrix::from_slice(&[inputs]);
    for i in 0..layers_count {
        a = a.dot(&m.w[i]).add(&m.b[i]).sigmoid();
    }

    a[0].to_vec()
}

pub fn cost(m: &ModelParams, train_in: &[&[f32]], train_out: &[&[f32]]) -> f32 {
    let output_neurons_count = m.w.last().unwrap().col_count();

    assert_eq!(train_in.len(), train_out.len());
    assert_eq!(train_out[0].len(), output_neurons_count);

    let samples_count = train_in.len();

    let mut cost = 0.0;
    for i in 0..samples_count {
        let x = train_in[i];
        let y = train_out[i];
        let guessed_y = forward(m, x);

        for j in 0..output_neurons_count {
            let diff = guessed_y[j] - y[j];
            cost += diff * diff;
        }
    }

    cost / (samples_count as f32)
}

pub fn finite_diff(
    m: &ModelParams,
    train_in: &[&[f32]],
    train_out: &[&[f32]],
    eps: f32,
) -> ModelParams {
    let c = cost(m, train_in, train_out);
    let layers_count = m.w.len();

    let mut grad = ModelParams::new(&m.structure());
    for i in 0..layers_count {
        let weights_of_layer = &m.w[i];
        for j in 0..weights_of_layer.row_count() {
            for k in 0..weights_of_layer.col_count() {
                let mut nudge = m.clone();
                nudge.w[i][j][k] += eps;
                grad.w[i][j][k] = (cost(&nudge, train_in, train_out) - c) / eps;
            }
        }

        let biases_of_layer = &m.b[i];
        for j in 0..biases_of_layer.row_count() {
            for k in 0..biases_of_layer.col_count() {
                let mut nudge = m.clone();
                nudge.b[i][j][k] += eps;
                grad.b[i][j][k] = (cost(&nudge, train_in, train_out) - c) / eps;
            }
        }
    }

    grad
}

pub fn descend(m: &ModelParams, grad: &ModelParams, rate: f32) -> ModelParams {
    let layers_count = m.w.len();

    let mut result = m.clone();
    for i in 0..layers_count {
        let weights_of_layer = &m.w[i];
        for j in 0..weights_of_layer.row_count() {
            for k in 0..weights_of_layer.col_count() {
                result.w[i][j][k] -= rate * grad.w[i][j][k];
            }
        }

        let biases_of_layer = &m.b[i];
        for j in 0..biases_of_layer.row_count() {
            for k in 0..biases_of_layer.col_count() {
                result.b[i][j][k] -= rate * grad.b[i][j][k];
            }
        }
    }

    result
}

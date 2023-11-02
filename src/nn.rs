use crate::matrix::Matrix;

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub struct ModelParams {
    pub w1: Matrix,
    pub b1: Matrix,
    pub w2: Matrix,
    pub b2: Matrix,
}

impl ModelParams {
    pub fn new() -> Self {
        Self {
            w1: Matrix::new(2, 2),
            b1: Matrix::new(1, 2),
            w2: Matrix::new(2, 1),
            b2: Matrix::new(1, 1),
        }
    }

    pub fn randomize(&mut self, low: f32, high: f32) {
        self.w1.randomize(low, high);
        self.b1.randomize(low, high);
        self.w2.randomize(low, high);
        self.b2.randomize(low, high);
    }
}

pub fn forward(m: &ModelParams, a0: &Matrix) -> f32 {
    let a1 = a0.dot(&m.w1).add(&m.b1).sigmoid();
    let a2 = a1.dot(&m.w2).add(&m.b2).sigmoid();

    // bad
    a2[0][0]
}

pub fn cost(m: &ModelParams, t_in: Matrix, t_out: Matrix) -> f32 {
    let output_neurons_count = m.w2.col_count();
    assert_eq!(t_in.row_count(), t_out.row_count());
    assert_eq!(t_out.col_count(), output_neurons_count);

    let sample_count = t_in.row_count();

    let mut cost = 0.0;
    for i in 0..sample_count {
        let x = Matrix::from_slice(&[&t_in[i]]);
        let y = Matrix::from_slice(&[&t_out[i]]);

        let guessed = forward(m, &x);

        for j in 0..output_neurons_count {
            let expected = y[0][j];
            let diff = guessed - expected;
            cost += diff * diff;
        }
    }

    cost / (sample_count as f32)
}

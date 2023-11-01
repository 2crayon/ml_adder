use std::{
    fmt,
    ops::{Index, IndexMut},
};

#[derive(Debug, Clone)]
pub struct Matrix {
    data: Vec<f32>,
    row_count: usize,
    col_count: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            row_count: rows,
            col_count: cols,
        }
    }

    pub fn fill(&mut self, value: f32) {
        for i in 0..self.data.len() {
            self.data[i] = value;
        }
    }

    pub fn randomize(&mut self, low: f32, high: f32) {
        for i in 0..self.data.len() {
            self.data[i] = rand::random::<f32>() * (high - low) + low;
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.row_count, other.row_count);
        assert_eq!(self.col_count, other.col_count);

        let mut result = Matrix::new(self.row_count, self.col_count);

        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }

        result
    }

    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(self.col_count, other.row_count);
        todo!()
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];

    fn index(&self, row: usize) -> &Self::Output {
        let start = row * self.col_count;
        let end = start + self.col_count;
        &self.data[start..end]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        let start = row * self.col_count;
        let end = start + self.col_count;
        &mut self.data[start..end]
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::new();
        for i in 0..self.row_count {
            s.push_str("|\t");
            for j in 0..self.col_count {
                s.push_str(&format!("{:.4}\t", self[i][j]));
            }
            s.push_str("\n");
        }

        write!(f, "{}", s)
    }
}

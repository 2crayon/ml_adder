mod matrix;

use matrix::Matrix;

static SEPERATOR: &str = "====================";

fn main() {
    let mut a = Matrix::new(2, 2);
    a.fill(1f32);

    let mut b = Matrix::new(2, 2);
    b.fill(1f32);

    println!("a:\n{}", a);
    println!("{}", SEPERATOR);
    println!("a+b\n{}", a.add(&b));
}

pub const EPSILON: f64 = 1e-09;
pub const INFINITY: f64 = f64::INFINITY;

pub fn random_f64() -> f64
{
    rand::random::<f64>()
}

pub fn random_cosine_vector() -> glam::DVec3
{
    let r1: f64 = random_f64();
    let r2: f64 = random_f64();
    let z: f64 = (1.0 - r2).sqrt();

    let phi: f64 = std::f64::consts::TAU * r1;
    let x: f64 = phi.cos() * r2.sqrt();
    let y: f64 = phi.sin() * r2.sqrt();

    glam::DVec3::new(x, y, z)
}

pub fn reflect(i: glam::DVec3, n: glam::DVec3) -> glam::DVec3
{
    i - 2.0 * glam::DVec3::dot(n, i) * n
}

pub fn refract(i: glam::DVec3, n: glam::DVec3, eta: f64) -> glam::DVec3
{
    let n_dot_i: f64 = glam::DVec3::dot(n, i);

    let k: f64 = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
    if k < 0.0 {
        glam::DVec3::new(0.0, 0.0, 0.0)
    } else {
        eta * i - (eta * n_dot_i + k.sqrt()) * n
    }
}

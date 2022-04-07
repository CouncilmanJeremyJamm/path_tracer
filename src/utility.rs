use nanorand::Rng;
use nanorand::tls::TlsWyRand;

pub const EPSILON: f32 = 5e-04;
pub const INFINITY: f32 = f32::INFINITY;

pub fn random_cosine_vector(rng: &mut TlsWyRand) -> glam::Vec3A
{
    let r1: f32 = rng.generate();
    let r2: f32 = rng.generate();
    let z: f32 = (1.0 - r2).sqrt();

    let phi: f32 = std::f32::consts::TAU * r1;
    let x: f32 = phi.cos() * r2.sqrt();
    let y: f32 = phi.sin() * r2.sqrt();

    glam::Vec3A::new(x, y, z)
}

pub fn reflect(i: glam::Vec3A, n: glam::Vec3A) -> glam::Vec3A { i - 2.0 * glam::Vec3A::dot(n, i) * n }

pub fn refract(i: glam::Vec3A, n: glam::Vec3A, eta: f32) -> glam::Vec3A
{
    let n_dot_i: f32 = glam::Vec3A::dot(n, i);

    let k: f32 = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
    if k < 0.0
    {
        glam::Vec3A::ZERO
    }
    else
    {
        eta * i - (eta * n_dot_i + k.sqrt()) * n
    }
}

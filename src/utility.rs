use nanorand::tls::TlsWyRand;
use nanorand::Rng;

pub const EPSILON: f32 = 5e-04;
pub const INFINITY: f32 = f32::INFINITY;

pub fn random_cosine_vector(rng: &mut TlsWyRand) -> glam::Vec3A
{
    let r: f32 = rng.generate::<f32>().sqrt();
    let z: f32 = (1.0 - r * r).sqrt();

    let phi: f32 = std::f32::consts::TAU * rng.generate::<f32>();
    let (sin, cos): (f32, f32) = phi.sin_cos();

    let x: f32 = cos * r;
    let y: f32 = sin * r;

    glam::Vec3A::new(x, y, z)
}

pub fn reflect(i: glam::Vec3A, n: glam::Vec3A) -> glam::Vec3A { i - 2.0 * glam::Vec3A::dot(n, i) * n }

pub fn refract(i: glam::Vec3A, n: glam::Vec3A, eta: f32) -> glam::Vec3A
{
    let n_dot_i: f32 = glam::Vec3A::dot(n, i);

    let k: f32 = 1.0 - eta * eta * (1.0 - n_dot_i * n_dot_i);
    if k <= 0.0
    {
        glam::Vec3A::NAN
    }
    else
    {
        eta * i - (eta * n_dot_i + k.sqrt()) * n
    }
}

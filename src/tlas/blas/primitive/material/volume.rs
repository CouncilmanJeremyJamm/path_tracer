use std::hash::{Hash, Hasher};

use glam::{Mat3A, Vec3A};
use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::tlas::blas::primitive::material::onb::generate_onb;

#[derive(Copy, Clone)]
pub struct VolumeScatter
{
    c: f32,
    g: f32,
}

impl VolumeScatter
{
    fn new(c: f32, g: f32) -> Self
    {
        Self {
            c,
            g: g.clamp(-0.999, 0.999),
        }
    }
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: Vec3A) -> Vec3A
    {
        let u0: f32 = rng.generate::<f32>();
        let u1: f32 = rng.generate::<f32>();

        let phi: f32 = 2.0 * std::f32::consts::PI * u0;

        let z: f32 = if self.g == 0.0
        {
            1.0 - 2.0 * u1
        }
        else
        {
            let x: f32 = (1.0 - self.g * self.g) / (1.0 + self.g * (1.0 - 2.0 * u1));
            (1.0 + self.g * self.g - x * x) / (2.0 * self.g)
        };

        let (sine, cosine): (f32, f32) = phi.sin_cos();

        let r: f32 = 1.0 - z * z;
        let x: f32 = r * cosine;
        let y: f32 = r * sine;

        let onb: Mat3A = generate_onb(-incoming.normalize());
        onb * glam::Vec3A::new(x, y, z)
    }

    fn scatter_pdf(&self, incoming: Vec3A, outgoing: Vec3A) -> f32
    {
        let wi: glam::Vec3A = outgoing.normalize();
        let wo: glam::Vec3A = incoming.normalize();

        let cosine: f32 = glam::Vec3A::dot(wi, wo);

        let n: f32 = 1.0 - self.g * self.g;
        let d: f32 = 4.0 * std::f32::consts::PI * (1.0 + self.g * self.g - 2.0 * self.g * cosine).powf(1.5);

        n / d
    }

    pub fn scatter(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, max_t: f32) -> Option<(f32, glam::Vec3A, f32)>
    {
        let t: f32 = -rng.generate::<f32>().ln() / self.c;
        if t > max_t
        {
            None
        }
        else
        {
            let outgoing: glam::Vec3A = self.scatter_direction(rng, incoming);
            let pdf: f32 = self.scatter_pdf(incoming, outgoing);

            Some((t, outgoing, pdf))
        }
    }
}

#[derive(Copy, Clone)]
pub struct VolumeAbsorption
{
    absorption: glam::Vec3A,
}

impl VolumeAbsorption
{
    pub fn new(absorption: glam::Vec3A, k: f32) -> Self { Self { absorption: absorption * k } }
    pub fn get_transmission(&self, dist: f32) -> glam::Vec3A { glam::Vec3A::exp(-self.absorption * dist) }
}

#[derive(Copy, Clone)]
pub struct Volume
{
    pub absorption: Option<VolumeAbsorption>,
    pub scatter: Option<VolumeScatter>,
}

impl Volume
{
    pub fn new(absorption: glam::Vec3A, k: f32, c: f32, g: f32) -> Self
    {
        let a = if k == 0.0 { None } else { Some(VolumeAbsorption::new(absorption, k)) };
        let s = if c == 0.0 { None } else { Some(VolumeScatter::new(c, g)) };

        Self { absorption: a, scatter: s }
    }
}

impl Hash for &Volume
{
    fn hash<H: Hasher>(&self, hasher: &mut H)
    {
        let ptr = (*self) as *const Volume;
        hasher.write_usize(ptr as usize);
    }
}

impl nohash_hasher::IsEnabled for &Volume {}

impl PartialEq<Self> for &Volume
{
    fn eq(&self, other: &Self) -> bool { std::ptr::eq(*self, *other) }
}

impl Eq for &Volume {}

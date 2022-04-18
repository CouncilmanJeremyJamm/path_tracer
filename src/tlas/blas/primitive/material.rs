use enum_dispatch::enum_dispatch;
use glam::{Mat3A, Vec3A};
use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::ray::Ray;
use crate::tlas::blas::primitive::material::onb::{generate_onb, generate_onb_ggx};
use crate::tlas::blas::primitive::model::HitInfo;
use crate::utility::{random_cosine_vector, reflect, refract};
use crate::Volume;

mod onb;
pub mod volume;

pub struct BsdfPdf
{
    pub bsdf: glam::Vec3A,
    pub pdf: f32,
}

impl BsdfPdf
{
    pub fn new(bsdf: glam::Vec3A, pdf: f32) -> Self
    {
        //debug_assert!(pdf > 0.0); //Not always true
        Self { bsdf, pdf }
    }
}

#[enum_dispatch]
pub trait MaterialTrait: Sync + Send
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, front_facing: bool) -> glam::Vec3A;
    fn get_brdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf;
    fn get_emitted(&self) -> glam::Vec3A { glam::Vec3A::ZERO }
    fn volume_scatter(&self, r: Ray, _: f32) -> (bool, glam::Vec3A) { (false, r.direction) }

    fn is_delta(&self) -> bool { false }
    fn is_emissive(&self) -> bool { false }
    fn get_volume(&self) -> Option<&Volume> { None }

    fn get_weakening(&self, wo: glam::Vec3A, n: glam::Vec3A) -> f32
    {
        if self.is_delta()
        {
            1.0
        }
        else
        {
            glam::Vec3A::dot(glam::Vec3A::normalize(wo), n).abs()
        }
    }
}

#[enum_dispatch(MaterialTrait)]
pub enum Material
{
    Lambertian,
    Emissive,
    Specular,
    GGXMetal,
    GGXDielectric,
    Dielectric,
}

pub struct Lambertian
{
    albedo: glam::Vec3A,
}

impl Lambertian
{
    pub fn new(albedo: glam::Vec3A) -> Material { Self { albedo }.into() }
}

impl MaterialTrait for Lambertian
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, _: glam::Vec3A, normal: glam::Vec3A, _: bool) -> glam::Vec3A
    {
        generate_onb(normal) * random_cosine_vector(rng)
    }

    fn get_brdf_pdf(&self, _incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        let cosine: f32 = glam::Vec3A::dot(glam::Vec3A::normalize(outgoing), hi.normal);
        let bsdf: glam::Vec3A = self.albedo * std::f32::consts::FRAC_1_PI;
        let pdf: f32 = cosine * std::f32::consts::FRAC_1_PI;
        BsdfPdf::new(bsdf, pdf)
    }
}

pub struct Emissive
{
    emitted: glam::Vec3A,
}

impl Emissive
{
    pub fn new(emitted: glam::Vec3A) -> Material { Self { emitted }.into() }
}

impl MaterialTrait for Emissive
{
    fn scatter_direction(&self, _: &mut TlsWyRand, _: glam::Vec3A, _: glam::Vec3A, _: bool) -> glam::Vec3A { glam::Vec3A::ZERO }

    fn get_brdf_pdf(&self, _: glam::Vec3A, _: glam::Vec3A, _: &HitInfo) -> BsdfPdf { BsdfPdf::new(self.emitted, 1.0) }

    fn get_emitted(&self) -> glam::Vec3A { self.emitted }

    fn is_emissive(&self) -> bool { true }
}

pub struct Specular
{
    colour: glam::Vec3A,
}

impl Specular
{
    pub fn new(colour: glam::Vec3A) -> Material { Self { colour }.into() }
}

impl MaterialTrait for Specular
{
    fn scatter_direction(&self, _: &mut TlsWyRand, incoming: Vec3A, normal: Vec3A, _: bool) -> Vec3A { reflect(incoming.normalize(), normal) }

    fn get_brdf_pdf(&self, _: Vec3A, _: Vec3A, _: &HitInfo) -> BsdfPdf { BsdfPdf::new(self.colour, 1.0) }

    fn is_delta(&self) -> bool { true }
}

struct GGX {}
impl GGX
{
    fn d(h: glam::Vec3A, a: f32) -> f32
    {
        if h.z <= 0.0
        {
            return 0.0;
        }

        let cosine_sq: f32 = h.z * h.z;
        let tan_sq: f32 = (1.0 - cosine_sq).sqrt() / cosine_sq;

        let x: f32 = (a * a) + tan_sq;
        a * a / (std::f32::consts::PI * cosine_sq * cosine_sq * x * x)
    }

    fn generate_half_vector(rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, a: f32) -> glam::Vec3A
    {
        let onb_a: glam::Mat3A = generate_onb(normal);

        let _v: glam::Vec3A = onb_a.transpose() * -incoming;
        let v: glam::Vec3A = (_v * glam::Vec3A::new(a, a, 1.0)).normalize();

        //Can't use generate_onb() for this, use implementation from the paper instead
        let onb_b: glam::Mat3A = generate_onb_ggx(v);

        let u1: f32 = rng.generate();
        let u2: f32 = rng.generate();

        let _a: f32 = 1.0 / (1.0 + v.z);
        let condition: bool = u2 < _a; //If condition is true, sample from the tilted half-disk

        let r: f32 = u1.sqrt();
        let phi: f32 = if condition
        {
            std::f32::consts::PI * u2 / _a
        }
        else
        {
            std::f32::consts::PI + ((u2 - _a) / (1.0 - _a)) * std::f32::consts::PI
        };
        let (sin, cos): (f32, f32) = phi.sin_cos();
        let p1: f32 = r * cos;
        let p2: f32 = r * sin * if condition { 1.0 } else { v.z };

        let _h: glam::Vec3A = onb_b * glam::Vec3A::new(p1, p2, (1.0 - p1 * p1 - p2 * p2).sqrt());

        onb_a * (_h * glam::Vec3A::new(a, a, 1.0)).normalize()
    }
}

pub struct GGXMetal
{
    colour: glam::Vec3A,
    a: f32,
}

impl GGXMetal
{
    fn f(&self, v_dot_h: f32) -> glam::Vec3A { self.colour + ((1.0 - self.colour) * (1.0 - v_dot_h).powi(5)) }

    fn g(&self, wi: glam::Vec3A, wo: glam::Vec3A) -> f32
    {
        if wi.z <= 0.0 || wo.z <= 0.0
        {
            return 0.0;
        }

        let a_squared: f32 = self.a * self.a;

        let x: f32 = 2.0 * wi.z * wo.z;
        let y: f32 = 1.0 - a_squared;
        let z: f32 = wo.z * (a_squared + (y * wi.z * wi.z)).sqrt();
        let w: f32 = wi.z * (a_squared + (y * wo.z * wo.z)).sqrt();

        x / (z + w)
    }
    pub fn new(colour: glam::Vec3A, roughness: f32) -> Material
    {
        Self {
            colour,
            a: roughness.powi(2).clamp(0.0001, 1.0),
        }
        .into()
    }
}

impl MaterialTrait for GGXMetal
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, _front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();
        let h: glam::Vec3A = GGX::generate_half_vector(rng, direction, normal, self.a);

        reflect(direction, h)
    }

    fn get_brdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        let onb_inv: Mat3A = generate_onb(hi.normal).transpose();

        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of scattering
        let wi: glam::Vec3A = onb_inv * outgoing.normalize();
        let wo: glam::Vec3A = onb_inv * incoming.normalize();

        let h: glam::Vec3A = (wi + wo).normalize();
        let d: f32 = GGX::d(h, self.a);
        let i_dot_h: f32 = glam::Vec3A::dot(wi, h);

        let num: glam::Vec3A = self.f(i_dot_h) * self.g(wi, wo) * d;
        let denom: f32 = 4.0 * wi.z * wo.z;

        let brdf: glam::Vec3A = num / denom;
        let pdf: f32 = d * h.z / (4.0 * glam::Vec3A::dot(wo, h));

        BsdfPdf::new(brdf, pdf)
    }
}

pub struct GGXDielectric
{
    colour: glam::Vec3A,
    ior: f32,
    a: f32,

    volume: Option<Volume>,
}

impl GGXDielectric
{
    fn f(&self, v_dot_h: f32, f0: f32) -> f32
    {
        if v_dot_h.is_finite()
        {
            f0 + ((1.0 - f0) * (1.0 - v_dot_h).powi(5))
        }
        else
        {
            1.0
        }
    }

    fn g_separable(&self, v: glam::Vec3A, h: glam::Vec3A) -> f32
    {
        if v.z * glam::Vec3A::dot(h, v) <= 0.0
        {
            return 0.0;
        }

        let n_dot_v_sq: f32 = v.z * v.z;
        let tan_squared: f32 = (1.0 - n_dot_v_sq) / n_dot_v_sq;
        2.0 / (1.0 + (1.0 + self.a * self.a * tan_squared).sqrt())
    }

    fn g(&self, wi: glam::Vec3A, wo: glam::Vec3A, h: glam::Vec3A) -> f32 { self.g_separable(wi, h) * self.g_separable(wo, h) }

    pub fn new(colour: glam::Vec3A, ior: f32, roughness: f32, volume: Option<Volume>) -> Material
    {
        Self {
            colour,
            ior,
            a: roughness.powi(2).clamp(0.0001, 0.9999),
            volume,
        }
        .into()
    }
}

impl MaterialTrait for GGXDielectric
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: Vec3A, normal: Vec3A, front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();

        //Generate half-vector from the GGX distribution
        let h: glam::Vec3A = GGX::generate_half_vector(rng, direction, normal, self.a);

        if -glam::Vec3A::dot(h, direction) < 1e-10
        {
            return glam::Vec3A::NAN;
        }

        //Reflect or refract using the half-vector as the normal
        let eta: f32 = if front_facing { self.ior.recip() } else { self.ior };
        let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        let refracted: glam::Vec3A = refract(direction, h, eta);

        //TODO: Fresnel term uses the refracted direction
        //Source: https://agraphicsguynotes.com/posts/glass_material_simulated_by_microfacet_bxdf/
        let f: f32 = self.f(-glam::Vec3A::dot(direction, h), f0);

        //println!("{f}");

        //If refract() returns NaN, this indicates total internal reflection
        let ray_reflected: bool = refracted.is_nan() || rng.generate::<f32>() < f;
        if ray_reflected
        {
            reflect(direction, h)
        }
        else
        {
            refracted
        }
    }

    fn get_brdf_pdf(&self, incoming: Vec3A, outgoing: Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        let onb_inv: Mat3A = generate_onb(hi.normal).transpose();

        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of scattering
        let wi: glam::Vec3A = onb_inv * outgoing.normalize();
        let wo: glam::Vec3A = onb_inv * incoming.normalize();

        let eta: f32 = if hi.front_facing { self.ior } else { self.ior.recip() };
        let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        let reflected: bool = wo.z * wi.z > 0.0;
        let _h: glam::Vec3A = if reflected
        {
            glam::Vec3A::normalize(wi + wo)
        }
        else
        {
            (eta * wi + wo).normalize()
        };

        let h: glam::Vec3A = _h * _h.z.signum();

        let i_dot_h: f32 = glam::Vec3A::dot(wi, h);
        let o_dot_h: f32 = glam::Vec3A::dot(wo, h);

        //let tir: bool = (1.0 - o_dot_h * o_dot_h) / (eta * eta) >= 1.0;

        let d: f32 = GGX::d(h, self.a);
        //TODO: Schlick's approximation uses the refracted direction
        //let f: f32 = if tir { 1.0 } else { self.f(i_dot_h.abs(), f0) };
        let f: f32 = self.f(i_dot_h.abs(), f0);
        let g: f32 = self.g(wi, wo, h);

        if reflected
        {
            //Calculate reflection BRDF
            let brdf: f32 = f * g * d / (4.0 * (wi.z * wo.z).abs());

            //Calculate PDF
            let jacobian: f32 = 1.0 / (4.0 * o_dot_h.abs());
            let pdf: f32 = d * h.z * f * jacobian;

            //Reflections are not affected by material colour
            BsdfPdf::new(glam::Vec3A::splat(brdf), pdf)
        }
        else
        {
            //Calculate transmission BSDF
            let x: f32 = (i_dot_h * o_dot_h).abs();
            let y: f32 = (wi.z * wo.z).abs();

            let z: f32 = (1.0 - f) * g * d;
            let w: f32 = (eta * i_dot_h) + o_dot_h;

            let btdf: f32 = (x * z) / (y * w * w);

            //Calculate PDF
            let ja: f32 = o_dot_h.abs();
            let jb: f32 = w;
            let jacobian: f32 = ja / (jb * jb);
            let pdf: f32 = d * (1.0 - f) * h.z.abs() * jacobian;

            //println!("{pdf}");

            //Transmission is affected by material colour
            BsdfPdf::new(self.colour * btdf * eta * eta, pdf)
        }
    }

    fn get_volume(&self) -> Option<&Volume> { self.volume.as_ref() }
}

//TODO: fix fresnel in dielectric
pub struct Dielectric
{
    colour: glam::Vec3A,
    ior: f32,

    volume: Option<Volume>,
}

impl Dielectric
{
    pub fn new(colour: glam::Vec3A, ior: f32, volume: Option<Volume>) -> Material { Self { colour, ior, volume }.into() }

    fn f(cosine: f32, eta: f32) -> f32
    {
        if eta * eta * (1.0 - cosine * cosine) > 1.0
        {
            1.0
        }
        else
        {
            let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);
            f0 + ((1.0 - f0) * (1.0 - cosine).powi(5))
        }
    }
}

impl MaterialTrait for Dielectric
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: Vec3A, normal: Vec3A, front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();

        let eta: f32 = if front_facing { self.ior.recip() } else { self.ior };
        let cosine: f32 = -glam::Vec3A::dot(direction, normal);

        if rng.generate::<f32>() < Self::f(cosine, eta)
        {
            reflect(direction, normal)
        }
        else
        {
            refract(direction, normal, eta)
        }
    }

    fn get_brdf_pdf(&self, incoming: Vec3A, outgoing: Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        let cosine: f32 = -glam::Vec3A::dot(incoming, outgoing);
        let eta: f32 = if hi.front_facing { self.ior.recip() } else { self.ior };
        let f: f32 = Self::f(cosine, eta);

        if glam::Vec3A::dot(outgoing, hi.normal) > 0.0
        {
            BsdfPdf::new(glam::Vec3A::splat(f), f)
        }
        else
        {
            //Account for solid-angle compression in refraction
            let bsdf: f32 = (1.0 - f) / (eta * eta);
            BsdfPdf::new(self.colour * bsdf, 1.0 - f)
        }
    }

    fn get_volume(&self) -> Option<&Volume> { self.volume.as_ref() }

    fn is_delta(&self) -> bool { true }
}

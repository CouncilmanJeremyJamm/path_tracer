use glam::Vec3A;

use crate::onb::{generate_onb, generate_onb_ggx};
use crate::primitive::model::HitInfo;
use crate::random_f32;
use crate::ray::Ray;
use crate::utility::{random_cosine_vector, reflect, refract};

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

pub trait Material
{
    fn scatter_direction(&self, incoming: glam::Vec3A, normal: glam::Vec3A, front_facing: bool) -> glam::Vec3A;
    fn get_brdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf;
    fn get_emitted(&self) -> glam::Vec3A { glam::Vec3A::ZERO }
    fn get_transmission(&self, _r: Ray, _t: f32) -> glam::Vec3A { glam::Vec3A::ZERO }
    fn volume_scatter(&self, r: Ray, _t: f32) -> (bool, glam::Vec3A) { (false, r.direction) }

    fn is_delta(&self) -> bool { false }
    fn is_emissive(&self) -> bool { false }

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

pub struct Lambertian
{
    albedo: glam::Vec3A,
}

impl Lambertian
{
    pub fn new(albedo: glam::Vec3A) -> Self { Self { albedo } }
}

impl Material for Lambertian
{
    fn scatter_direction(&self, _incoming: glam::Vec3A, normal: glam::Vec3A, _front_facing: bool) -> glam::Vec3A
    {
        generate_onb(normal) * random_cosine_vector()
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
    pub fn new(emitted: glam::Vec3A) -> Self { Self { emitted } }
}

impl Material for Emissive
{
    fn scatter_direction(&self, _: glam::Vec3A, _: glam::Vec3A, _: bool) -> glam::Vec3A { glam::Vec3A::ZERO }

    fn get_brdf_pdf(&self, _: glam::Vec3A, _: glam::Vec3A, _: &HitInfo) -> BsdfPdf { BsdfPdf::new(self.emitted, 1.0) }

    fn get_emitted(&self) -> glam::Vec3A { self.emitted }

    fn is_emissive(&self) -> bool { true }
}

#[allow(clippy::upper_case_acronyms)]
struct GGX {}
impl GGX
{
    fn d(n: glam::Vec3A, h: glam::Vec3A, a: f32) -> f32
    {
        let n_dot_h: f32 = glam::Vec3A::dot(n, h);
        let x: f32 = (n_dot_h * n_dot_h * (a * a - 1.0)) + 1.0;

        a * a * std::f32::consts::FRAC_1_PI / (x * x)
    }

    fn generate_half_vector(incoming: glam::Vec3A, normal: glam::Vec3A, a: f32) -> glam::Vec3A
    {
        let onb_a: glam::Mat3A = generate_onb(normal);

        let _v: glam::Vec3A = onb_a.transpose() * -incoming;
        let v: glam::Vec3A = (_v * glam::Vec3A::new(a, a, 1.0)).normalize();

        //Can't use generate_onb() for this, use implementation from the paper instead
        let onb_b: glam::Mat3A = generate_onb_ggx(v);

        let u1: f32 = random_f32();
        let u2: f32 = random_f32();

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

//TODO: pack colour and a
pub struct GGX_Metal
{
    colour: glam::Vec3A,
    a: f32,
}

#[allow(clippy::upper_case_acronyms)]
impl GGX_Metal
{
    fn f(&self, v_dot_h: f32) -> glam::Vec3A { self.colour + ((1.0 - self.colour) * (1.0 - v_dot_h).powi(5)) }

    fn g(&self, n: glam::Vec3A, wi: glam::Vec3A, wo: glam::Vec3A) -> f32
    {
        let n_dot_i: f32 = glam::Vec3A::dot(n, wi);
        let n_dot_o: f32 = glam::Vec3A::dot(n, wo);

        if n_dot_i <= 0.0 || n_dot_o <= 0.0
        {
            return 0.0;
        }

        let a_squared: f32 = self.a * self.a;

        let x: f32 = 2.0 * n_dot_i * n_dot_o;
        let y: f32 = 1.0 - a_squared;
        let z: f32 = n_dot_o * (a_squared + (y * n_dot_i * n_dot_i)).sqrt();
        let w: f32 = n_dot_i * (a_squared + (y * n_dot_o * n_dot_o)).sqrt();

        x / (z + w)
    }
    pub fn new(colour: glam::Vec3A, roughness: f32) -> Self
    {
        Self {
            colour,
            a: roughness.powi(2).clamp(0.0001, 1.0),
        }
    }
}

impl Material for GGX_Metal
{
    fn scatter_direction(&self, incoming: glam::Vec3A, normal: glam::Vec3A, _front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();
        let h: glam::Vec3A = GGX::generate_half_vector(direction, normal, self.a);

        reflect(direction, h)
    }

    //TODO: simplify. Precalculate dot products and pass to functions
    fn get_brdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of reflection
        let wi: glam::Vec3A = outgoing.normalize();
        let wo: glam::Vec3A = incoming.normalize();

        let h: glam::Vec3A = (wi + wo).normalize();
        let d: f32 = GGX::d(hi.normal, h, self.a);
        let o_dot_h: f32 = glam::Vec3A::dot(wo, h);

        let num: glam::Vec3A = self.f(o_dot_h) * self.g(hi.normal, wi, wo) * d;
        let denom: f32 = 4.0 * glam::Vec3A::dot(hi.normal, wi) * glam::Vec3A::dot(hi.normal, wo);

        let brdf: glam::Vec3A = num / denom;
        let pdf: f32 = d * glam::Vec3A::dot(h, hi.normal) / (4.0 * glam::Vec3A::dot(wo, h));

        BsdfPdf::new(brdf, pdf)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct GGX_Dielectric
{
    absorption: glam::Vec3A,
    colour: glam::Vec3A,
    ior: f32,
    a: f32,
}

impl GGX_Dielectric
{
    fn f(&self, v_dot_h: f32, f0: f32) -> f32 { f0 + ((1.0 - f0) * (1.0 - v_dot_h).powi(5)) }

    fn g_separable(&self, v: glam::Vec3A, h: glam::Vec3A, n: glam::Vec3A) -> f32
    {
        let n_dot_v: f32 = glam::Vec3A::dot(n, v);

        if n_dot_v / glam::Vec3A::dot(h, v) <= 0.0
        {
            return 0.0;
        }

        let n_dot_v_sq: f32 = n_dot_v * n_dot_v;
        let tan_squared: f32 = (1.0 - n_dot_v_sq) / n_dot_v_sq;
        2.0 / (1.0 + (1.0 + (self.a * self.a * tan_squared)).sqrt())
    }

    fn g(&self, wi: glam::Vec3A, wo: glam::Vec3A, h: glam::Vec3A, n: glam::Vec3A) -> f32 { self.g_separable(wi, h, n) * self.g_separable(wo, h, n) }

    pub fn new(absorption: glam::Vec3A, colour: glam::Vec3A, ior: f32, a: f32) -> Self { Self { absorption, colour, ior, a } }
}

impl Material for GGX_Dielectric
{
    fn scatter_direction(&self, incoming: Vec3A, normal: Vec3A, front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();

        //Generate half-vector from the GGX distribution
        let h: glam::Vec3A = GGX::generate_half_vector(direction, normal, self.a);

        //Reflect or refract using the half-vector as the normal
        let cosine: f32 = -glam::Vec3A::dot(direction, h);
        let sine: f32 = (1.0 - cosine * cosine).sqrt();

        let eta: f32 = if front_facing { 1.0 / self.ior } else { self.ior };
        let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        let tir: bool = eta * sine > 1.0;

        //Total internal reflection
        if tir
        {
            return reflect(direction, h);
        }

        //Fresnel term must be calculated from the refracted ray, not the incident ray
        //Source: https://agraphicsguynotes.com/posts/glass_material_simulated_by_microfacet_bxdf/
        let refracted: glam::Vec3A = refract(direction, h, eta);

        if random_f32() < self.f(-glam::Vec3A::dot(refracted, h), f0)
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
        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of scattering
        let wi: glam::Vec3A = outgoing.normalize();
        let wo: glam::Vec3A = incoming.normalize();

        let eta: f32 = if hi.front_facing { self.ior } else { 1.0 / self.ior };
        let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        if glam::Vec3A::dot(wi, hi.normal) > 0.0
        {
            //Incident ray reflected
            let h: glam::Vec3A = glam::Vec3A::normalize(wi + wo);

            //Calculate reflection BRDF
            //Schlick's approximation uses the refracted direction
            let o_dot_h: f32 = glam::Vec3A::dot(wo, h);
            let r_dot_h: f32 = glam::Vec3A::dot(refract(-wo, h, 1.0 / eta), h);

            let d: f32 = GGX::d(hi.normal, h, self.a);
            let f: f32 = self.f(-r_dot_h, f0);

            let u: f32 = f * self.g(wi, wo, h, hi.normal) * d;
            let v: f32 = 4.0 * glam::Vec3A::dot(hi.normal, wi).abs() * glam::Vec3A::dot(hi.normal, wo);

            let brdf: f32 = u / v;

            //Calculate PDF
            let jacobian: f32 = 1.0 / (4.0 * o_dot_h.abs());
            let pdf: f32 = d * glam::Vec3A::dot(h, hi.normal) * f * jacobian;

            //Reflections are not affected by material colour
            BsdfPdf::new(glam::Vec3A::new(brdf, brdf, brdf), pdf)
        }
        else
        {
            //Incident ray refracted
            let _h: glam::Vec3A = ((eta * wi) + wo).normalize();
            let h: glam::Vec3A = if glam::Vec3A::dot(hi.normal, _h) > 0.0 { _h } else { -_h };
            let d: f32 = GGX::d(hi.normal, h, self.a);

            let i_dot_h: f32 = glam::Vec3A::dot(wi, h);
            let o_dot_h: f32 = glam::Vec3A::dot(wo, h);

            //Calculate transmission BSDF
            let x: f32 = (i_dot_h * o_dot_h).abs();
            let y: f32 = (glam::Vec3A::dot(wi, hi.normal) * glam::Vec3A::dot(wo, hi.normal)).abs();
            //TODO: fix i dot h
            let f: f32 = self.f(-i_dot_h, f0);

            let z: f32 = (1.0 - f) * self.g(wi, wo, h, hi.normal) * d;
            let w: f32 = (eta * i_dot_h) + o_dot_h;

            let btdf: f32 = (x * z) / (y * w * w);

            //Calculate PDF
            let ja: f32 = o_dot_h.abs();
            let jb: f32 = w;
            let jacobian: f32 = ja / (jb * jb);
            let pdf: f32 = d * (1.0 - f) * glam::Vec3A::dot(h, hi.normal) * jacobian;
            //println!("{}", f);
            //Transmission is affected by material colour
            BsdfPdf::new(self.colour * btdf, pdf)
        }
    }
}

pub struct Dielectric
{
    colour: glam::Vec3A,
    ior: f32,
}

impl Dielectric
{
    pub fn new(colour: glam::Vec3A, ior: f32) -> Self { Self { colour, ior } }

    fn f(cosine: f32, eta: f32) -> f32
    {
        let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        f0 + ((1.0 - f0) * (1.0 - cosine).powi(5))
    }
}

impl Material for Dielectric
{
    fn scatter_direction(&self, incoming: Vec3A, normal: Vec3A, front_facing: bool) -> glam::Vec3A
    {
        let direction: glam::Vec3A = incoming.normalize();
        let cosine: f32 = -glam::Vec3A::dot(direction, normal);
        let sine: f32 = (1.0 - cosine * cosine).sqrt();

        let eta: f32 = if front_facing { self.ior.recip() } else { self.ior };
        let must_reflect: bool = eta * sine > 1.0;

        if must_reflect || random_f32() < Self::f(cosine, eta)
        {
            reflect(direction, normal)
        }
        else
        {
            refract(direction, normal, eta)
        }
    }

    fn get_brdf_pdf(&self, _: Vec3A, outgoing: Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        if glam::Vec3A::dot(outgoing, hi.normal) > 0.0
        {
            BsdfPdf::new(glam::Vec3A::new(1.0, 1.0, 1.0), 1.0)
        }
        else
        {
            BsdfPdf::new(self.colour, 1.0)
        }
    }

    fn is_delta(&self) -> bool { true }
}

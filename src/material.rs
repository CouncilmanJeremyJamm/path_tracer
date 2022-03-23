use crate::onb::{generate_onb, generate_onb_ggx};
use crate::primitive::model::HitInfo;
use crate::random_f64;
use crate::ray::Ray;
use crate::utility::{random_cosine_vector, reflect, refract};
use glam::DVec3;

pub struct BsdfPdf
{
    pub bsdf: glam::DVec3,
    pub pdf: f64,
}

impl BsdfPdf
{
    pub fn new(bsdf: glam::DVec3, pdf: f64) -> Self
    {
        //debug_assert!(pdf > 0.0); //Not always true
        Self { bsdf, pdf }
    }
}

pub trait Material
{
    fn scatter_direction(
        &self,
        incoming: glam::DVec3,
        normal: glam::DVec3,
        front_facing: bool,
    ) -> glam::DVec3;
    fn get_brdf_pdf(&self, incoming: glam::DVec3, outgoing: glam::DVec3, hi: &HitInfo) -> BsdfPdf;
    fn get_emitted(&self) -> glam::DVec3
    {
        glam::DVec3::new(0.0, 0.0, 0.0)
    }
    fn get_transmission(&self, _r: Ray, _t: f64) -> glam::DVec3
    {
        glam::DVec3::new(0.0, 0.0, 0.0)
    }
    fn volume_scatter(&self, r: Ray, _t: f64) -> (bool, glam::DVec3)
    {
        (false, r.direction)
    }

    fn is_delta(&self) -> bool
    {
        false
    }
    fn is_emissive(&self) -> bool
    {
        false
    }

    fn get_weakening(&self, wo: glam::DVec3, n: glam::DVec3) -> f64
    {
        if self.is_delta() {
            1.0
        } else {
            glam::DVec3::dot(glam::DVec3::normalize(wo), n).abs()
        }
    }
}

pub struct Lambertian
{
    albedo: glam::DVec3,
}

impl Lambertian
{
    pub fn new(albedo: glam::DVec3) -> Self
    {
        Self { albedo }
    }
}

impl Material for Lambertian
{
    fn scatter_direction(
        &self,
        _incoming: glam::DVec3,
        normal: glam::DVec3,
        _front_facing: bool,
    ) -> glam::DVec3
    {
        generate_onb(normal) * random_cosine_vector()
    }

    fn get_brdf_pdf(&self, _incoming: glam::DVec3, outgoing: glam::DVec3, hi: &HitInfo) -> BsdfPdf
    {
        let cosine: f64 = glam::DVec3::dot(glam::DVec3::normalize(outgoing), hi.normal);
        let bsdf: glam::DVec3 = self.albedo * std::f64::consts::FRAC_1_PI;
        let pdf: f64 = cosine * std::f64::consts::FRAC_1_PI;
        BsdfPdf::new(bsdf, pdf)
    }
}

pub struct Emissive
{
    emitted: glam::DVec3,
}

impl Emissive
{
    pub fn new(emitted: glam::DVec3) -> Self
    {
        Self { emitted }
    }
}

impl Material for Emissive
{
    fn scatter_direction(
        &self,
        _incoming: glam::DVec3,
        _normal: glam::DVec3,
        _front_facing: bool,
    ) -> glam::DVec3
    {
        glam::DVec3::new(0.0, 0.0, 0.0)
    }

    fn get_brdf_pdf(&self, _incoming: glam::DVec3, _outgoing: glam::DVec3, _hi: &HitInfo)
        -> BsdfPdf
    {
        BsdfPdf::new(self.emitted, 1.0)
    }

    fn get_emitted(&self) -> glam::DVec3
    {
        self.emitted
    }

    fn is_emissive(&self) -> bool
    {
        true
    }
}

struct GGX {}
impl GGX
{
    fn d(n: glam::DVec3, h: glam::DVec3, a: f64) -> f64
    {
        let n_dot_h: f64 = glam::DVec3::dot(n, h);
        let x: f64 = (n_dot_h * n_dot_h * (a * a - 1.0)) + 1.0;

        a * a * std::f64::consts::FRAC_1_PI / (x * x)
    }

    fn generate_half_vector(incoming: glam::DVec3, normal: glam::DVec3, a: f64) -> glam::DVec3
    {
        let onb_a: glam::DMat3 = generate_onb(normal);

        let _v: glam::DVec3 = onb_a.transpose() * -incoming;
        let v: glam::DVec3 = (_v * glam::DVec3::new(a, a, 1.0)).normalize();

        //Can't use generate_onb() for this, use implementation from the paper instead
        let onb_b: glam::DMat3 = generate_onb_ggx(v);

        let u1: f64 = random_f64();
        let u2: f64 = random_f64();

        let _a: f64 = 1.0 / (1.0 + v.z);
        let condition: bool = u2 < _a; //If condition is true, sample from the tilted half-disk

        let r: f64 = u1.sqrt();
        let phi: f64 = if condition {
            std::f64::consts::PI * u2 / _a
        } else {
            std::f64::consts::PI + ((u2 - _a) / (1.0 - _a)) * std::f64::consts::PI
        };
        let (sin, cos): (f64, f64) = phi.sin_cos();
        let p1: f64 = r * cos;
        let p2: f64 = r * sin * if condition { 1.0 } else { v.z };

        let _h: glam::DVec3 = onb_b * glam::DVec3::new(p1, p2, (1.0 - p1 * p1 - p2 * p2).sqrt());

        onb_a * (_h * glam::DVec3::new(a, a, 1.0)).normalize()
    }
}

//TODO: pack colour and a
pub struct GGX_Metal
{
    colour: glam::DVec3,
    a: f64,
}

#[allow(clippy::upper_case_acronyms)]
impl GGX_Metal
{
    fn f(&self, v_dot_h: f64) -> glam::DVec3
    {
        self.colour + ((1.0 - self.colour) * (1.0 - v_dot_h).powi(5))
    }

    fn g(&self, n: glam::DVec3, wi: glam::DVec3, wo: glam::DVec3) -> f64
    {
        let n_dot_i: f64 = glam::DVec3::dot(n, wi);
        let n_dot_o: f64 = glam::DVec3::dot(n, wo);

        if n_dot_i <= 0.0 || n_dot_o <= 0.0 {
            return 0.0;
        }

        let a_squared: f64 = self.a * self.a;

        let x: f64 = 2.0 * n_dot_i * n_dot_o;
        let y: f64 = 1.0 - a_squared;
        let z: f64 = n_dot_o * (a_squared + (y * n_dot_i * n_dot_i)).sqrt();
        let w: f64 = n_dot_i * (a_squared + (y * n_dot_o * n_dot_o)).sqrt();

        x / (z + w)
    }
    pub fn new(colour: glam::DVec3, roughness: f64) -> Self
    {
        Self {
            colour,
            a: roughness.clamp(0.0001, 1.0),
        }
    }
}

impl Material for GGX_Metal
{
    fn scatter_direction(
        &self,
        incoming: glam::DVec3,
        normal: glam::DVec3,
        _front_facing: bool,
    ) -> glam::DVec3
    {
        let direction: glam::DVec3 = incoming.normalize();
        let h: glam::DVec3 = GGX::generate_half_vector(direction, normal, self.a);

        reflect(direction, h)
    }

    //TODO: simplify. Precalculate dot products and pass to functions
    fn get_brdf_pdf(&self, incoming: glam::DVec3, outgoing: glam::DVec3, hi: &HitInfo) -> BsdfPdf
    {
        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of reflection
        let wi: glam::DVec3 = outgoing.normalize();
        let wo: glam::DVec3 = incoming.normalize();

        let h: glam::DVec3 = (wi + wo).normalize();
        let d: f64 = GGX::d(hi.normal, h, self.a);
        let o_dot_h: f64 = glam::DVec3::dot(wo, h);

        let num: glam::DVec3 = self.f(o_dot_h) * self.g(hi.normal, wi, wo) * d;
        let denom: f64 = 4.0 * glam::DVec3::dot(hi.normal, wi) * glam::DVec3::dot(hi.normal, wo);

        let brdf: glam::DVec3 = num / denom;
        let pdf: f64 = d * glam::DVec3::dot(h, hi.normal) / (4.0 * glam::DVec3::dot(wo, h));

        BsdfPdf::new(brdf, pdf)
    }
}

#[allow(clippy::upper_case_acronyms)]
pub struct GGX_Dielectric
{
    absorption: glam::DVec3,
    colour: glam::DVec3,
    ior: f64,
    a: f64,
}

impl GGX_Dielectric
{
    fn f(&self, v_dot_h: f64, f0: f64) -> f64
    {
        f0 + ((1.0 - f0) * (1.0 - v_dot_h).powi(5))
    }

    fn g_separable(&self, v: glam::DVec3, h: glam::DVec3, n: glam::DVec3) -> f64
    {
        let n_dot_v: f64 = glam::DVec3::dot(n, v);

        if n_dot_v / glam::DVec3::dot(h, v) <= 0.0 {
            return 0.0;
        }

        let n_dot_v_sq: f64 = n_dot_v * n_dot_v;
        let tan_squared: f64 = (1.0 - n_dot_v_sq) / n_dot_v_sq;
        2.0 / (1.0 + (1.0 + (self.a * self.a * tan_squared)).sqrt())
    }

    fn g(&self, wi: glam::DVec3, wo: glam::DVec3, h: glam::DVec3, n: glam::DVec3) -> f64
    {
        self.g_separable(wi, h, n) * self.g_separable(wo, h, n)
    }

    pub fn new(absorption: glam::DVec3, colour: glam::DVec3, ior: f64, a: f64) -> Self
    {
        Self {
            absorption,
            colour,
            ior,
            a,
        }
    }
}

impl Material for GGX_Dielectric
{
    fn scatter_direction(&self, incoming: DVec3, normal: DVec3, front_facing: bool) -> glam::DVec3
    {
        let direction: glam::DVec3 = incoming.normalize();

        //Generate half-vector from the GGX distribution
        let h: glam::DVec3 = GGX::generate_half_vector(direction, normal, self.a);

        //Reflect or refract using the half-vector as the normal
        let cosine: f64 = -glam::DVec3::dot(direction, h);
        let sine: f64 = (1.0 - cosine * cosine).sqrt();

        let eta: f64 = if front_facing {
            1.0 / self.ior
        } else {
            self.ior
        };
        let f0: f64 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        let tir: bool = eta * sine > 1.0;

        //Total internal reflection
        if tir {
            return reflect(direction, h);
        }

        //Fresnel term must be calculated from the refracted ray, not the incident ray
        //Source: https://agraphicsguynotes.com/posts/glass_material_simulated_by_microfacet_bxdf/
        let refracted: glam::DVec3 = refract(direction, h, eta);

        if random_f64() < self.f(-glam::DVec3::dot(refracted, h), f0) {
            reflect(direction, h)
        } else {
            refracted
        }
    }

    fn get_brdf_pdf(&self, incoming: DVec3, outgoing: DVec3, hi: &HitInfo) -> BsdfPdf
    {
        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of scattering
        let wi: glam::DVec3 = outgoing.normalize();
        let wo: glam::DVec3 = incoming.normalize();

        let eta: f64 = if hi.front_facing {
            self.ior
        } else {
            1.0 / self.ior
        };
        let f0: f64 = ((eta - 1.0) / (eta + 1.0)).powi(2);

        if glam::DVec3::dot(wi, hi.normal) > 0.0 {
            //Incident ray reflected
            let h: glam::DVec3 = glam::DVec3::normalize(wi + wo);

            //Calculate reflection BRDF
            //Schlick's approximation uses the refracted direction
            let o_dot_h: f64 = glam::DVec3::dot(wo, h);
            let r_dot_h: f64 = glam::DVec3::dot(refract(-wo, h, 1.0 / eta), h);

            let d: f64 = GGX::d(hi.normal, h, self.a);
            let f: f64 = self.f(-r_dot_h, f0);

            let u: f64 = f * self.g(wi, wo, h, hi.normal) * d;
            let v: f64 =
                4.0 * glam::DVec3::dot(hi.normal, wi).abs() * glam::DVec3::dot(hi.normal, wo);

            let brdf: f64 = u / v;

            //Calculate PDF
            let jacobian: f64 = 1.0 / (4.0 * o_dot_h.abs());
            let pdf: f64 = d * glam::DVec3::dot(h, hi.normal) * f * jacobian;

            //Reflections are not affected by material colour
            BsdfPdf::new(glam::DVec3::new(brdf, brdf, brdf), pdf)
        } else {
            //Incident ray refracted
            let _h: glam::DVec3 = ((eta * wi) + wo).normalize();
            let h: glam::DVec3 = if glam::DVec3::dot(hi.normal, _h) > 0.0 {
                _h
            } else {
                -_h
            };
            let d: f64 = GGX::d(hi.normal, h, self.a);

            let i_dot_h: f64 = glam::DVec3::dot(wi, h);
            let o_dot_h: f64 = glam::DVec3::dot(wo, h);

            //Calculate transmission BSDF
            let x: f64 = (i_dot_h * o_dot_h).abs();
            let y: f64 = (glam::DVec3::dot(wi, hi.normal) * glam::DVec3::dot(wo, hi.normal)).abs();
            //TODO: fix i dot h
            let f: f64 = self.f(-i_dot_h, f0);

            let z: f64 = (1.0 - f) * self.g(wi, wo, h, hi.normal) * d;
            let w: f64 = (eta * i_dot_h) + o_dot_h;

            let btdf: f64 = (x * z) / (y * w * w);

            //Calculate PDF
            let ja: f64 = o_dot_h.abs();
            let jb: f64 = w;
            let jacobian: f64 = ja / (jb * jb);
            let pdf: f64 = d * (1.0 - f) * glam::DVec3::dot(h, hi.normal) * jacobian;
            println!("{}", f);
            //Transmission is affected by material colour
            BsdfPdf::new(self.colour * btdf, pdf)
        }
    }
}

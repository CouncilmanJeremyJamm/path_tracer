use ambassador::delegatable_trait;
use ambassador::Delegate;
use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::tlas::tlas_bvh::blas::primitive::material::onb::{generate_onb, generate_onb_ggx};
use crate::tlas::tlas_bvh::blas::primitive::model::HitInfo;
use crate::utility::{random_cosine_vector, reflect, refract};
use crate::Volume;

mod onb;
pub mod volume;

/// Return type for `MaterialTrait::get_bsdf_pdf()`
pub struct BsdfPdf
{
    pub bsdf: glam::Vec3A,
    pub pdf: f32,
}

impl BsdfPdf
{
    pub fn new(bsdf: glam::Vec3A, pdf: f32) -> Self { Self { bsdf, pdf } }
    pub fn new_delta(bsdf: glam::Vec3A) -> Self { Self { bsdf, pdf: 1.0 } }

    pub fn invalid() -> Self
    {
        Self {
            bsdf: glam::Vec3A::ZERO,
            pdf: 0.0,
        }
    }
}

#[delegatable_trait]
/// Common interface for all material types.
///
/// Implements `Sync + Send` to allow multithreading
pub trait MaterialTrait: Sync + Send
{
    const DELTA: bool = false;
    const EMISSIVE: bool = false;

    /// Generates an outgoing direction
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, front_facing: bool) -> glam::Vec3A;

    /// For given incoming and outgoing directions, calculate both the BSDF and the pdf
    fn get_bsdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf;

    /// Returns the light emitted by the current material. Only non-zero for the `Emissive` material
    fn get_emitted(&self) -> glam::Vec3A { glam::Vec3A::ZERO }

    /// Returns `true` if the current material has a delta distribution
    fn is_delta(&self) -> bool { Self::DELTA }

    /// Returns `true` if the current material type is `Emissive`
    fn is_emissive(&self) -> bool { Self::EMISSIVE }

    /// Returns `None` for non-transmissive materials.
    ///
    /// May return `Some(&Volume)` if the current material is both transmissive and has volume attributes.
    fn get_volume(&self) -> Option<&Volume> { None }

    /// Calculates the cosine term in the light transport equation.
    ///
    /// For delta distributions, this returns 1.0.
    fn get_weakening(&self, wo: glam::Vec3A, n: glam::Vec3A) -> f32
    {
        if self.is_delta()
        {
            1.0
        }
        else
        {
            glam::Vec3A::dot(wo, n).abs()
        }
    }
}

#[derive(Clone, Delegate)]
#[delegate(MaterialTrait)]
pub enum Material
{
    Lambertian(Lambertian),
    Emissive(Emissive),
    Specular(Specular),
    GGX(GGX),
    Dielectric(Dielectric),
}

#[derive(Clone)]
pub struct Lambertian
{
    albedo: glam::Vec3A,
}

impl Lambertian
{
    pub fn new(albedo: glam::Vec3A) -> Material { Material::Lambertian(Self { albedo }) }
}

impl MaterialTrait for Lambertian
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, _: glam::Vec3A, normal: glam::Vec3A, _: bool) -> glam::Vec3A
    {
        generate_onb(normal) * random_cosine_vector(rng)
    }

    fn get_bsdf_pdf(&self, _incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        let cosine: f32 = glam::Vec3A::dot(outgoing, hi.normal);
        let bsdf: glam::Vec3A = self.albedo * std::f32::consts::FRAC_1_PI;
        let pdf: f32 = cosine * std::f32::consts::FRAC_1_PI;
        BsdfPdf::new(bsdf, pdf)
    }
}

#[derive(Clone)]
pub struct Emissive
{
    emitted: glam::Vec3A,
}

impl Emissive
{
    pub fn new(emitted: glam::Vec3A) -> Material { Material::Emissive(Self { emitted }) }
}

impl MaterialTrait for Emissive
{
    const EMISSIVE: bool = true;

    fn scatter_direction(&self, _: &mut TlsWyRand, _: glam::Vec3A, _: glam::Vec3A, _: bool) -> glam::Vec3A { glam::Vec3A::ZERO }
    fn get_bsdf_pdf(&self, _: glam::Vec3A, _: glam::Vec3A, _: &HitInfo) -> BsdfPdf { BsdfPdf::new_delta(self.emitted) }
    fn get_emitted(&self) -> glam::Vec3A { self.emitted }
}

#[derive(Clone)]
pub struct Specular
{
    colour: glam::Vec3A,
}

impl Specular
{
    pub fn new(colour: glam::Vec3A) -> Material { Material::Specular(Self { colour }) }
}

impl MaterialTrait for Specular
{
    const DELTA: bool = true;

    fn scatter_direction(&self, _: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, _: bool) -> glam::Vec3A { reflect(incoming, normal) }

    fn get_bsdf_pdf(&self, _: glam::Vec3A, _: glam::Vec3A, _: &HitInfo) -> BsdfPdf { BsdfPdf::new_delta(self.colour) }
}

/// Implementation of GGX model based on *Microfacet Models for Refraction through Rough Surfaces*

#[derive(Clone)]
pub struct GGX
{
    /// Surface colour
    colour: glam::Vec3A,
    /// Surface roughness, remapped from linear roughness
    a: f32,
    /// Indicates which model is used, dictating which scatter directions are valid
    ggx_model: GGXModel,
}

/// The `GGXModel` enum allows the `GGX` struct to implement two separate models:
/// * `REFLECTIVE` - metals. Only reflection is enabled
/// * `TRANSMISSIVE` - dielectrics. Both reflection and refraction is enabled

#[derive(Clone)]
enum GGXModel
{
    REFLECTIVE,
    TRANSMISSIVE
    {
        volume: Option<Volume>,
        ior: f32,
    },
}

impl GGX
{
    /// Normal distribution function
    fn d(&self, h: glam::Vec3A) -> f32
    {
        if h.z <= 0.0
        {
            return 0.0;
        }

        let cosine_sq: f32 = h.z * h.z;
        let tan_sq: f32 = (1.0 - cosine_sq).sqrt() / cosine_sq;

        let x: f32 = (self.a * self.a) + tan_sq;
        self.a * self.a / (std::f32::consts::PI * cosine_sq * cosine_sq * x * x)
    }

    /// Schlick's approximation to the fresnel term
    // fn f(&self, v_dot_h: f32, f0: f32) -> f32 { f0 + ((1.0 - f0) * (1.0 - v_dot_h).powi(5)) }
    fn f(&self, v_dot_h: f32, f0: f32) -> f32 { (1.0 - v_dot_h).powi(5).mul_add(1.0 - f0, f0) }
    /// Schlick's approximation to the fresnel term, used to tint the reflection for the `REFLECTIVE` model
    fn f_vector(&self, v_dot_h: f32, f0: glam::Vec3A) -> glam::Vec3A { f0 + ((1.0 - f0) * (1.0 - v_dot_h).powi(5)) }

    /// Mono-directional/masking shadowing function
    fn g1(&self, v: glam::Vec3A, h: glam::Vec3A) -> f32
    {
        if v.z * glam::Vec3A::dot(h, v) <= 0.0
        {
            return 0.0;
        }

        // let n_dot_v_sq: f32 = v.z * v.z;
        // let tan_squared: f32 = (1.0 - n_dot_v_sq) / n_dot_v_sq;
        let tan_squared: f32 = v.z.powi(-2) - 1.0;
        2.0 / (1.0 + (1.0 + self.a * self.a * tan_squared).sqrt())
    }

    /// Bi-directional/masking-shadowing shadowing function
    fn g(&self, wi: glam::Vec3A, wo: glam::Vec3A, h: glam::Vec3A) -> f32 { self.g1(wi, h) * self.g1(wo, h) }

    /// Uncorrelated `G` term based on *Moving Frostbite to Physically Based Rendering*
    fn g_uncorrelated(&self, wi: glam::Vec3A, wo: glam::Vec3A) -> f32
    {
        if wi.z <= 0.0 || wo.z <= 0.0
        {
            return 0.0;
        }

        let a_squared: f32 = self.a * self.a;

        let x: f32 = 2.0 * wi.z * wo.z;
        let y: f32 = 1.0 - a_squared;
        // let z: f32 = wo.z * (a_squared + (y * wi.z * wi.z)).sqrt();
        // let w: f32 = wi.z * (a_squared + (y * wo.z * wo.z)).sqrt();
        let z: f32 = wo.z * f32::hypot(self.a, wi.z * y.sqrt());
        let w: f32 = wi.z * f32::hypot(self.a, wo.z * y.sqrt());

        x / (z + w)
    }

    /// Samples the half-vector using the method outlined in
    /// *A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals*
    fn generate_half_vector(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A) -> glam::Vec3A
    {
        // Transform into tangent space before using the sampling method
        let onb_a: glam::Mat3A = generate_onb(normal);

        let _v: glam::Vec3A = onb_a.transpose() * -incoming;
        let v: glam::Vec3A = (_v * glam::Vec3A::new(self.a, self.a, 1.0)).normalize();

        //Can't use generate_onb() for this, use implementation from the paper instead
        let onb_b: glam::Mat3A = generate_onb_ggx(v);

        let u1: f32 = rng.generate();
        let u2: f32 = rng.generate();

        let _a: f32 = 1.0 / (1.0 + v.z);
        let condition: bool = u2 < _a; //If condition is true, sample from the tilted half-disk

        // If r == 1.0, the resultant half-vector contains only NaN components
        let r: f32 = u1.sqrt().min(0.9999);
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

        // Un-stretch the half-vector and transform back to world space
        onb_a * (_h * glam::Vec3A::new(self.a, self.a, 1.0)).normalize()
    }

    /// Constructs a GGX material using the `REFLECTIVE` model
    /// # Parameters
    /// * `colour` - surface colour
    /// * `roughness` - linear surface roughness
    pub fn new_metal(colour: glam::Vec3A, roughness: f32) -> Material
    {
        Material::GGX(Self {
            colour,
            a: roughness.powi(2).clamp(0.0001, 0.9999),
            ggx_model: GGXModel::REFLECTIVE,
        })
    }

    /// Constructs a GGX material using the `REFRACTIVE` model
    /// # Parameters
    /// * `colour` - surface colour
    /// * `roughness` - linear surface roughness
    /// * `absorption` - absorption due to Beer-Lambert law
    /// * `ior` - index of refraction
    pub fn new_dielectric(colour: glam::Vec3A, roughness: f32, ior: f32, volume: Option<Volume>) -> Material
    {
        Material::GGX(Self {
            colour,
            a: roughness.powi(2).clamp(0.0001, 0.9999),
            ggx_model: GGXModel::TRANSMISSIVE { ior, volume },
        })
    }
}

impl MaterialTrait for GGX
{
    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, front_facing: bool) -> glam::Vec3A
    {
        //Generate half-vector from the GGX distribution
        let h: glam::Vec3A = self.generate_half_vector(rng, incoming, normal);

        //Reflect or refract using the half-vector as the normal
        match self.ggx_model
        {
            GGXModel::REFLECTIVE => reflect(incoming, h),
            GGXModel::TRANSMISSIVE { ior, .. } =>
            {
                let eta: f32 = if front_facing { ior.recip() } else { ior };
                let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);

                let f: f32 = self.f(-glam::Vec3A::dot(incoming, h), f0);

                //If refract() returns NaN, this indicates total internal reflection
                let refracted: glam::Vec3A = refract(incoming, h, eta);
                let ray_reflected: bool = refracted.is_nan() || rng.generate::<f32>() < f;

                if ray_reflected
                {
                    reflect(incoming, h)
                }
                else
                {
                    refracted
                }
            }
        }
    }

    fn get_bsdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
    {
        //Transform to tangent space
        let onb_inv: glam::Mat3A = generate_onb(hi.normal).transpose();

        //Outgoing = direction of light ray = -direction of tracing ray
        //Incoming = direction of scattering
        let wi: glam::Vec3A = onb_inv * outgoing;
        let wo: glam::Vec3A = onb_inv * incoming;

        let ray_transmitted: bool = wi.z < 0.0;

        let h: glam::Vec3A = match self.ggx_model
        {
            GGXModel::REFLECTIVE => glam::Vec3A::normalize(wi + wo),
            GGXModel::TRANSMISSIVE { ior, .. } =>
            {
                if ray_transmitted
                {
                    let eta: f32 = if hi.front_facing { ior } else { ior.recip() };
                    let _h: glam::Vec3A = (eta * wi + wo).normalize();
                    _h * _h.z.signum()
                }
                else
                {
                    glam::Vec3A::normalize(wi + wo)
                }
            }
        };

        let i_dot_h: f32 = glam::Vec3A::dot(wi, h);
        let o_dot_h: f32 = glam::Vec3A::dot(wo, h);

        let d: f32 = self.d(h);

        let (f, g): (f32, f32) = match self.ggx_model
        {
            //The reflective model must reflect, and f == 1.0
            GGXModel::REFLECTIVE => (1.0, self.g_uncorrelated(wi, wo)),
            GGXModel::TRANSMISSIVE { ior, .. } =>
            {
                let eta: f32 = if hi.front_facing { ior } else { ior.recip() };

                let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);
                let _f = self.f(i_dot_h.abs(), f0);
                let _g = self.g(wi, wo, h);

                (_f, _g)
            }
        };

        if ray_transmitted
        {
            match self.ggx_model
            {
                //Return early for illegal ray directions
                GGXModel::REFLECTIVE => BsdfPdf::invalid(),
                GGXModel::TRANSMISSIVE { ior, .. } =>
                {
                    //Calculate transmission BSDF
                    let eta: f32 = if hi.front_facing { ior } else { ior.recip() };

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

                    //Transmission is affected by surface colour
                    BsdfPdf::new(self.colour * btdf * eta * eta, pdf)
                }
            }
        }
        else
        {
            //Both metal and dielectric share a similar BRDF/pdf for reflection
            //Calculate reflection BRDF
            let brdf: f32 = f * g * d / (4.0 * (wi.z * wo.z).abs());

            //Calculate PDF
            let jacobian: f32 = 1.0 / (4.0 * o_dot_h.abs());
            let pdf: f32 = d * h.z * f * jacobian;

            //Reflections in the reflective model must be tinted for colour to appear
            //Reflections in the transmissive model are not tinted
            let reflection_tint: glam::Vec3A = match self.ggx_model
            {
                GGXModel::REFLECTIVE => self.f_vector(i_dot_h.abs(), self.colour),
                GGXModel::TRANSMISSIVE { .. } => glam::Vec3A::ONE,
            };

            BsdfPdf::new(brdf * reflection_tint, pdf)
        }
    }

    fn get_volume(&self) -> Option<&Volume>
    {
        match &self.ggx_model
        {
            GGXModel::REFLECTIVE => None,
            GGXModel::TRANSMISSIVE { volume, .. } => volume.as_ref(),
        }
    }
}

//TODO: fix fresnel in dielectric

#[derive(Clone)]
pub struct Dielectric
{
    colour: glam::Vec3A,
    ior: f32,

    volume: Option<Volume>,
}

impl Dielectric
{
    pub fn new(colour: glam::Vec3A, ior: f32, volume: Option<Volume>) -> Material { Material::Dielectric(Self { colour, ior, volume }) }

    fn f(cosine: f32, eta: f32) -> f32
    {
        if eta * eta * (1.0 - cosine * cosine) > 1.0
        {
            1.0
        }
        else
        {
            let f0: f32 = ((eta - 1.0) / (eta + 1.0)).powi(2);
            // f0 + ((1.0 - f0) * (1.0 - cosine).powi(5))
            (1.0 - cosine).powi(5).mul_add(1.0 - f0, f0)
        }
    }
}

impl MaterialTrait for Dielectric
{
    const DELTA: bool = true;

    fn scatter_direction(&self, rng: &mut TlsWyRand, incoming: glam::Vec3A, normal: glam::Vec3A, front_facing: bool) -> glam::Vec3A
    {
        let eta: f32 = if front_facing { self.ior.recip() } else { self.ior };
        let cosine: f32 = -glam::Vec3A::dot(incoming, normal);

        if rng.generate::<f32>() < Self::f(cosine, eta)
        {
            reflect(incoming, normal)
        }
        else
        {
            refract(incoming, normal, eta)
        }
    }

    fn get_bsdf_pdf(&self, incoming: glam::Vec3A, outgoing: glam::Vec3A, hi: &HitInfo) -> BsdfPdf
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
}

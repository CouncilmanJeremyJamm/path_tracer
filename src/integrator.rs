use bumpalo::Bump;
use image::ImageError;
use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use nohash_hasher::IntSet;

use crate::image_helper::ImageHelper;
use crate::{BsdfPdf, HitInfo, Material, MaterialTrait, Ray, Scene, Triangle, Volume, ENABLE_NEE, EPSILON, INFINITY};

const MIN_PDF: f32 = 0.0;
const HEURISTIC_POWER: i32 = 2;

#[inline]
/// Heuristic for generating MIS weights
/// # Parameters
/// * `f` - pdf of current sampling technique
/// * `g` - pdf of another sampling technique
/// * `power` - corresponds to different heuristic functions
///     * `0`: naive weighting, returns `0.5`
///     * `1`: balance heuristic
///     * `2`: power heuristic
fn mis_heuristic<const POWER: i32>(f: f32, g: f32) -> f32 { f.powi(POWER) / (f.powi(POWER) + g.powi(POWER)) }

/// Direct lighting estimation by explicit light sampling
fn estimate_direct_explicit(
    rng: &mut TlsWyRand,
    bump: &Bump,
    incoming_ray: &Ray,
    current_hi: &HitInfo,
    current_material: &Material,
    scene: &Scene,
) -> glam::Vec3A
{
    let incoming: glam::Vec3A = -incoming_ray.direction;

    // Pick a random light to sample using the LightSampler in Scene
    // Each light has a pdf proportional to area * ||emitted||
    let (light_material, light, pdf): (&Material, &Triangle, f32) = scene.sample_lights(rng);

    // Pick a random point on the light
    // Point is sampled uniformly with pdf = 1.0 / area
    let (point, light_normal): (glam::Vec3A, glam::Vec3A) = light.random_point(rng);

    // Construct ray towards light
    let o: glam::Vec3A = incoming_ray.at(current_hi.t);
    let d: glam::Vec3A = point - o;

    let distance_squared: f32 = d.length_squared();
    let distance: f32 = distance_squared.sqrt();

    let outgoing_ray: Ray = Ray::new(o, d.normalize());

    // Cast shadow ray
    // If light ray points against the normal, occlusion is guaranteed and the shadow ray is skipped
    if glam::Vec3A::dot(outgoing_ray.direction, current_hi.normal) > 0.0
        && !scene.world.any_intersect(bump, &outgoing_ray, (1.0 - EPSILON) * distance)
    {
        // Calculate pdf for current direction, given BSDF sampling
        let bsdf_pdf: BsdfPdf = current_material.get_bsdf_pdf(incoming, outgoing_ray.direction, current_hi);

        let sample_pdf: f32 = pdf / light.area();

        // Calculate pdf for current direction, given light sampling
        let cosine: f32 = glam::Vec3A::dot(outgoing_ray.direction, light_normal).abs();
        let light_pdf: f32 = sample_pdf * (distance_squared / cosine);

        // Calculate weight from heuristic, then accumulate direct light
        let weight: f32 = mis_heuristic::<HEURISTIC_POWER>(light_pdf, bsdf_pdf.pdf);
        return light_material.get_emitted() * weight * current_material.get_weakening(outgoing_ray.direction, current_hi.normal) * bsdf_pdf.bsdf
            / light_pdf;
    }

    glam::Vec3A::ZERO
}

/// Direct light estimation by sampling the bsdf
fn estimate_direct_bsdf(
    rng: &mut TlsWyRand,
    bump: &Bump,
    incoming_ray: &Ray,
    current_hi: &HitInfo,
    current_material: &Material,
    scene: &Scene,
) -> glam::Vec3A
{
    let incoming: glam::Vec3A = -incoming_ray.direction;

    //Sample ray from BSDF
    let outgoing_ray: Ray = Ray::new(
        incoming_ray.at(current_hi.t),
        current_material.scatter_direction(rng, incoming_ray.direction, current_hi.normal, current_hi.front_facing),
    );

    // Cast material ray
    // If material ray points against the normal, occlusion is guaranteed and the shadow ray is skipped
    if glam::Vec3A::dot(outgoing_ray.direction, current_hi.normal) > 0.0
    {
        // Cheap intersection test with BVH of lights only
        // Checks for a potential light intersection without traversing the full scene BVH
        if let Some((light_hi, _, light, light_material)) = scene.lights.intersect(bump, &outgoing_ray, INFINITY)
        {
            // Cheap test passed, do full shadow ray test
            if !scene.world.any_intersect(bump, &outgoing_ray, light_hi.t * (1.0 - EPSILON))
            {
                // Calculate pdf for current direction, given BSDF sampling
                let bsdf_pdf: BsdfPdf = current_material.get_bsdf_pdf(incoming, outgoing_ray.direction, current_hi);

                if bsdf_pdf.pdf > MIN_PDF
                {
                    // Get pdf for choosing the intersected light
                    let sample_pdf: f32 = scene.light_sampler.get_sample_pdf(light, light_material) / light.area();

                    // Calculate pdf for current direction, given light sampling
                    let cosine: f32 = glam::Vec3A::dot(outgoing_ray.direction, light_hi.normal).abs();
                    let light_pdf: f32 = sample_pdf * (light_hi.t * light_hi.t / cosine);

                    // Calculate weight from heuristic, then accumulate direct light
                    let weight: f32 = mis_heuristic::<HEURISTIC_POWER>(bsdf_pdf.pdf, light_pdf);
                    return light_material.get_emitted()
                        * weight
                        * current_material.get_weakening(outgoing_ray.direction, current_hi.normal)
                        * bsdf_pdf.bsdf
                        / bsdf_pdf.pdf;
                }
            }
        }
    }

    glam::Vec3A::ZERO
}

/// Direct lighting estimation using Multiple-Importance Sampling (MIS).
/// Rays are generated using two methods:
/// 1.   Sampled from the material BSDF
/// 2.   Uniformly sampling a point on a light
///
/// The outputs are then weighted using a heuristic function.
fn estimate_direct(rng: &mut TlsWyRand, bump: &Bump, r: &Ray, hit_info: &HitInfo, mat: &Material, scene: &Scene) -> glam::Vec3A
{
    estimate_direct_explicit(rng, bump, r, hit_info, mat, scene) + estimate_direct_bsdf(rng, bump, r, hit_info, mat, scene)
}

pub(crate) fn integrate(
    mut r: Ray,
    scene: &Scene,
    env: &Result<ImageHelper, ImageError>,
    rng: &mut TlsWyRand,
    max_bounces: u32,
) -> (glam::Vec4, glam::Vec4, u8)
{
    let bump: Bump = Bump::new();

    let mut accumulated: glam::Vec3A = glam::Vec3A::ZERO;
    let mut path_weight: glam::Vec3A = glam::Vec3A::ONE;

    let mut position: glam::Vec4 = r.at(1e5).extend(1e5);
    let mut first_id: u8 = u8::MAX;

    let mut last_delta: bool = false;

    let mut volume_stack: IntSet<&Volume> = IntSet::default();

    'trace_ray: for b in 0..=max_bounces
    {
        //Russian roulette
        if b > 3
        {
            let survive_prob: f32 = path_weight.max_element().min(0.9999);
            if rng.generate::<f32>() > survive_prob
            {
                break 'trace_ray;
            }
            else
            {
                path_weight /= survive_prob;
            }
        }

        if let Some((hit_info, id, _, material)) = scene.world.intersect(&bump, &r, INFINITY)
        {
            if b == 0
            {
                position = r.at(hit_info.t).extend(hit_info.t);
                first_id = id.index() as u8;
            }

            let wi: glam::Vec3A = -r.direction;

            let absorbing = volume_stack.iter().filter_map(|v| v.absorption.as_ref());
            let scattering = volume_stack.iter().filter_map(|v| v.scatter.as_ref());

            if let Some((t, dir, _)) = scattering
                .filter_map(|s| s.scatter(rng, r.direction, hit_info.t))
                .min_by(|a, b| a.0.total_cmp(&b.0))
            {
                path_weight *= absorbing.fold(glam::Vec3A::ONE, |w, a| w * a.get_transmission(t));
                last_delta = true;

                r = Ray::new(r.at(t), dir);
                continue 'trace_ray;
            }
            else
            {
                path_weight *= absorbing.fold(glam::Vec3A::ONE, |w, a| w * a.get_transmission(hit_info.t));
            }

            if material.is_emissive()
            {
                if !ENABLE_NEE || last_delta || b == 0
                {
                    accumulated = material.get_emitted().mul_add(path_weight, accumulated);
                }
                break 'trace_ray;
            }
            else
            {
                if let Some(v) = material.get_volume()
                {
                    if hit_info.front_facing
                    {
                        volume_stack.insert(v);
                    }
                    else
                    {
                        volume_stack.remove(&v);
                    }
                }

                let is_delta: bool = material.is_delta();

                if ENABLE_NEE && !is_delta
                {
                    accumulated += path_weight * estimate_direct(rng, &bump, &r, &hit_info, material, scene);
                }

                r = Ray::new(
                    r.at(hit_info.t),
                    material.scatter_direction(rng, r.direction, hit_info.normal, hit_info.front_facing),
                );

                let material_info: BsdfPdf = material.get_bsdf_pdf(wi, r.direction, &hit_info);

                if material_info.pdf < MIN_PDF
                {
                    //println!("Invalid PDF, returning early");
                    break 'trace_ray;
                }

                path_weight *= material.get_weakening(r.direction, hit_info.normal) * material_info.bsdf / material_info.pdf;

                last_delta = is_delta;
            }
        }
        else
        {
            if let Ok(image) = env.as_ref()
            {
                let u: f32 = r.direction.x.atan2(r.direction.z).mul_add(std::f32::consts::FRAC_1_PI * 0.5, 0.5);
                let v: f32 = r.direction.y.asin().mul_add(-std::f32::consts::FRAC_1_PI, 0.5);

                accumulated += image.get_pixel_bilinear(u, v) * path_weight;
            }
            else
            {
                accumulated += glam::Vec3A::new(0.006, 0.006, 0.006) * path_weight;
            }

            break 'trace_ray;
        }
    }

    if accumulated.is_finite()
    {
        (accumulated.clamp_length_max(100.0).extend(1.0), position, first_id)
        // accumulated.extend(1.0)
    }
    else
    {
        (glam::Vec4::W, position, first_id)
    }
}

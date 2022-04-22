use bumpalo::Bump;
use image::{DynamicImage, GenericImageView, ImageResult};
use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use nohash_hasher::IntSet;

use crate::{BsdfPdf, HitInfo, Material, MaterialTrait, Ray, Scene, Triangle, Volume, ENABLE_NEE, EPSILON, INFINITY};

const MIN_PDF: f32 = 1e-8;

#[inline]
fn power_heuristic(f: f32, g: f32) -> f32 { (f * f) / (f * f + g * g) }

/// Direct lighting estimation using Multiple-Importance Sampling (MIS).
/// Rays are generated using two methods:
/// 1.   Sampled from the material BSDF
/// 2.   Uniformly sampling a point on a light
fn estimate_direct(rng: &mut TlsWyRand, bump: &Bump, r: &Ray, hit_info: &HitInfo, mat: &Material, scene: &Scene) -> glam::Vec3A
{
    let mut direct: glam::Vec3A = glam::Vec3A::ZERO;
    let incoming: glam::Vec3A = -r.direction;

    // Pick a random light to sample using the LightSampler in Scene
    // Each light has a pdf proportional to area * ||emitted||
    let (light_material, light, pdf): (&Material, &Triangle, f32) = scene.sample_lights(rng);

    // Pick a random point on the light
    // Point is sampled uniformly with pdf = 1.0 / area
    let (point, light_normal): (glam::Vec3A, glam::Vec3A) = light.random_point(rng);

    // Construct ray towards light
    let o: glam::Vec3A = r.at(hit_info.t);
    let d: glam::Vec3A = point - o;

    let distance_squared: f32 = d.length_squared();
    let distance: f32 = distance_squared.sqrt();

    let light_ray: Ray = Ray::new(o, d / distance);

    // Cast shadow ray
    // If light ray points against the normal, occlusion is guaranteed and the shadow ray is skipped
    if glam::Vec3A::dot(light_ray.direction, hit_info.normal) > 0.0 && !scene.world.any_intersect(bump, &light_ray, (1.0 - EPSILON) * distance)
    {
        // Calculate pdf for current direction, given BSDF sampling
        let light_info: BsdfPdf = mat.get_bsdf_pdf(incoming, light_ray.direction, hit_info);

        if light_info.pdf > MIN_PDF
        {
            // Calculate pdf for current direction, given light sampling
            let cosine: f32 = glam::Vec3A::dot(light_ray.direction, light_normal).abs();
            let light_pdf: f32 = distance_squared * pdf / (cosine * light.area());

            // Calculate weight from heuristic, then accumulate direct light
            let weight: f32 = power_heuristic(light_pdf, light_info.pdf);
            direct += light_material.get_emitted() * weight * mat.get_weakening(light_ray.direction, hit_info.normal) * light_info.bsdf / light_pdf;
        }
    }

    //Sample ray from BSDF
    let material_ray: Ray = Ray::new(o, mat.scatter_direction(rng, r.direction, hit_info.normal, hit_info.front_facing));

    // Cast material ray
    // If material ray points against the normal, occlusion is guaranteed and the shadow ray is skipped
    if glam::Vec3A::dot(material_ray.direction, hit_info.normal) > 0.0
    {
        // Cheap intersection test with BVH of lights only
        // Checks for a potential light intersection without traversing the full scene BVH
        if let Some((material_hi, intersected, material)) = scene.lights.intersect(bump, &material_ray, INFINITY)
        {
            // Cheap test passed, do full shadow ray test
            if !scene.world.any_intersect(bump, &material_ray, material_hi.t * (1.0 - EPSILON))
            {
                // Calculate pdf for current direction, given BSDF sampling
                let material_info: BsdfPdf = mat.get_bsdf_pdf(incoming, material_ray.direction, hit_info);

                if material_info.pdf > MIN_PDF
                {
                    // Get pdf for choosing the intersected light
                    let p: f32 = scene.light_sampler.get_sample_pdf(intersected, material);

                    // Calculate pdf for current direction, given light sampling
                    let cosine: f32 = glam::Vec3A::dot(material_ray.direction, material_hi.normal).abs();
                    let light_pdf: f32 = material_hi.t * material_hi.t * p / (cosine * intersected.area());

                    // Calculate weight from heuristic, then accumulate direct light
                    let weight: f32 = power_heuristic(material_info.pdf, light_pdf);
                    direct += material.get_emitted() * weight * mat.get_weakening(material_ray.direction, hit_info.normal) * material_info.bsdf
                        / material_info.pdf;
                }
            }
        }
    }

    direct
}

pub(crate) fn integrate(mut r: Ray, scene: &Scene, env: &ImageResult<DynamicImage>, max_bounces: u32) -> glam::Vec3A
{
    let mut rng: TlsWyRand = nanorand::tls_rng();
    let bump: Bump = Bump::new();

    let mut accumulated: glam::Vec3A = glam::Vec3A::ZERO;
    let mut path_weight: glam::Vec3A = glam::Vec3A::ONE;

    let mut last_delta: bool = false;

    let mut volume_stack: IntSet<&Volume> = IntSet::default();

    for b in 0..=max_bounces
    {
        //Russian roulette
        if b > 3 || path_weight.max_element() < 1e-10
        {
            let survive_prob: f32 = path_weight.max_element().min(0.9999);
            if rng.generate::<f32>() > survive_prob
            {
                break;
            }
            else
            {
                path_weight /= survive_prob;
            }
        }

        if let Some((hit_info, _, material)) = scene.world.intersect(&bump, &r, INFINITY)
        {
            let wi: glam::Vec3A = -r.direction;

            let absorbing = volume_stack.iter().filter_map(|v| v.absorption.as_ref());
            let scattering = volume_stack.iter().filter_map(|v| v.scatter.as_ref());

            let (volume_scattered, dist) = if let Some((t, dir, _)) = scattering
                .filter_map(|s| s.scatter(&mut rng, r.direction, hit_info.t))
                .min_by(|a, b| a.0.total_cmp(&b.0))
            {
                r = Ray::new(r.at(t), dir);

                (true, r.direction.length() * t)
            }
            else
            {
                (false, r.direction.length() * hit_info.t)
            };

            path_weight *= absorbing.fold(glam::Vec3A::ONE, |w, a| w * a.get_transmission(dist));

            if volume_scattered
            {
                last_delta = true;
                continue;
            }

            if material.is_emissive()
            {
                if !ENABLE_NEE || last_delta || b == 0
                {
                    accumulated += material.get_emitted() * path_weight;
                }
                break;
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
                    accumulated += path_weight * estimate_direct(&mut rng, &bump, &r, &hit_info, material, scene);
                }

                r = Ray::new(
                    r.at(hit_info.t),
                    material.scatter_direction(&mut rng, r.direction, hit_info.normal, hit_info.front_facing),
                );

                let material_info: BsdfPdf = material.get_bsdf_pdf(wi, r.direction, &hit_info);

                if material_info.pdf < MIN_PDF
                {
                    //println!("Invalid PDF, returning early");
                    break;
                }

                path_weight *= material.get_weakening(r.direction, hit_info.normal) * material_info.bsdf / material_info.pdf;

                last_delta = is_delta;
            }
        }
        else
        {
            if env.is_ok()
            //if false
            {
                let image: &DynamicImage = env.as_ref().unwrap();
                let dimensions: (u32, u32) = image.dimensions();

                let u: f32 = r.direction.x.atan2(r.direction.z).mul_add(std::f32::consts::FRAC_1_PI * 0.5, 0.5);
                let v: f32 = r.direction.y.asin().mul_add(-std::f32::consts::FRAC_1_PI, 0.5);

                let x: f32 = (dimensions.0 as f32) * u;
                let y: f32 = (dimensions.1 as f32) * v;

                let x0: u32 = (x as u32) % dimensions.0;
                let y0: u32 = (y as u32) % dimensions.1;

                let x1: u32 = (x0 + 1) % dimensions.0;
                let y1: u32 = (y0 + 1) % dimensions.1;

                let x_fract: f32 = x.fract();
                let y_fract: f32 = y.fract();

                //TODO: move bi-linear interpolation into function
                let c_00: glam::Vec3A = pixel_to_vec3(image.get_pixel(x0, y0));
                let c_01: glam::Vec3A = pixel_to_vec3(image.get_pixel(x0, y1));
                let c_10: glam::Vec3A = pixel_to_vec3(image.get_pixel(x1, y0));
                let c_11: glam::Vec3A = pixel_to_vec3(image.get_pixel(x1, y1));

                let colour: glam::Vec3A = (1.0 - x_fract) * (1.0 - y_fract) * c_00
                    + (1.0 - x_fract) * y_fract * c_01
                    + x_fract * (1.0 - y_fract) * c_10
                    + x_fract * y_fract * c_11;

                accumulated += colour * path_weight;
            }
            else
            {
                accumulated += glam::Vec3A::new(0.006, 0.006, 0.006) * path_weight;
            }

            break;
        }
    }

    if accumulated.is_finite()
    {
        accumulated
    }
    else
    {
        glam::Vec3A::ZERO
    }
}

fn u8_to_float(a: u8) -> f32 { ((a as f32) / 255.0).powf(2.2) }

fn f32_to_u8(a: f32) -> u8 { (a.powf(1.0 / 2.2) * 255.0) as u8 }

pub fn map_colour(a: &glam::Vec3A) -> [u8; 3]
{
    [
        f32_to_u8(a.x.clamp(0.0, 1.0)),
        f32_to_u8(a.y.clamp(0.0, 1.0)),
        f32_to_u8(a.z.clamp(0.0, 1.0)),
    ]
}

fn pixel_to_vec3(p: image::Rgba<u8>) -> glam::Vec3A { glam::Vec3A::new(u8_to_float(p[0]), u8_to_float(p[1]), u8_to_float(p[2])) }

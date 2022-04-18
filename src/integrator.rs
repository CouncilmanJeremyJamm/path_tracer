use bumpalo::Bump;
use image::{DynamicImage, GenericImageView, ImageResult};
use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use nohash_hasher::IntSet;

use crate::{BsdfPdf, HitInfo, Material, MaterialTrait, Ray, Scene, Triangle, Volume, ENABLE_NEE, EPSILON, INFINITY};

const MIN_PDF: f32 = 1e-8;

#[allow(dead_code)]
#[inline]
fn balance_heuristic(f: f32, g: f32) -> f32 { f / (f + g) }

#[inline]
fn power_heuristic(f: f32, g: f32) -> f32 { (f * f) / (f * f + g * g) }

fn estimate_direct(rng: &mut TlsWyRand, bump: &Bump, r: &Ray, hit_info: &HitInfo, mat: &Material, scene: &Scene) -> glam::Vec3A
{
    let mut direct: glam::Vec3A = glam::Vec3A::ZERO;
    let incoming: glam::Vec3A = -r.direction;

    let (light_material, light, pdf): (&Material, &Triangle, f32) = scene.sample_lights(rng);
    //println!("{}", pdf);
    let (point, light_normal): (glam::Vec3A, glam::Vec3A) = light.random_point(rng);

    let o: glam::Vec3A = r.at(hit_info.t);
    let d: glam::Vec3A = point - o;

    let light_ray: Ray = Ray::new(o, d);

    if !scene.world.any_intersect(bump, &light_ray, 1.0 - EPSILON)
    {
        let light_info: BsdfPdf = mat.get_brdf_pdf(incoming, light_ray.direction, hit_info);

        //Special case, ray to light might give an invalid PDF
        //All other rays are generated by the material and must have a valid PDF by construction
        if light_info.pdf > MIN_PDF
        {
            let cosine: f32 = glam::Vec3A::dot(d.normalize(), light_normal).abs();
            let light_pdf: f32 = d.length_squared() * pdf / (cosine * light.area());

            let weight: f32 = power_heuristic(light_pdf, light_info.pdf);
            direct += light_material.get_emitted() * weight * mat.get_weakening(light_ray.direction, hit_info.normal) * light_info.bsdf / light_pdf;
        }
    }

    //Sample ray from BSDF
    let material_ray: Ray = Ray::new(o, mat.scatter_direction(rng, r.direction, hit_info.normal, hit_info.front_facing));

    if !material_ray.direction.is_nan()
    {
        if let Some((material_hi, intersected, material)) = scene.lights.intersect(bump, &material_ray, INFINITY)
        {
            if !scene.world.any_intersect(bump, &material_ray, material_hi.t * (1.0 - EPSILON))
            {
                let material_info: BsdfPdf = mat.get_brdf_pdf(incoming, material_ray.direction, hit_info);

                if material_info.pdf > MIN_PDF
                {
                    let p: f32 = scene.light_sampler.get_sample_pdf(intersected, material);

                    let cosine: f32 = glam::Vec3A::dot(material_ray.direction, material_hi.normal).abs();
                    let light_pdf: f32 = material_hi.t * material_hi.t * p / (cosine * intersected.area());

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

            if material.is_emissive() && (!ENABLE_NEE || last_delta || b == 0)
            {
                accumulated += material.get_emitted() * path_weight;
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

                if r.direction.is_nan()
                {
                    break;
                }

                let material_info: BsdfPdf = material.get_brdf_pdf(wi, r.direction, &hit_info);

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

                let dir: glam::Vec3A = glam::Vec3A::normalize(r.direction);

                let u: f32 = dir.x.atan2(dir.z).mul_add(std::f32::consts::FRAC_1_PI * 0.5, 0.5);
                let v: f32 = dir.y.asin().mul_add(-std::f32::consts::FRAC_1_PI, 0.5);

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

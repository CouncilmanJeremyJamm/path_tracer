#![allow(clippy::upper_case_acronyms, clippy::new_ret_no_self)]
#![feature(allocator_api, array_chunks, bool_to_option)]
extern crate core;

use rayon::prelude::*;

use tlas::blas::primitive::material::*;
use tlas::blas::primitive::model::{HitInfo, Model};
use tlas::blas::primitive::Triangle;

use crate::camera::Camera;
use crate::ray::Ray;
use crate::scene::Scene;
use crate::tlas::TLAS;
use crate::utility::{EPSILON, INFINITY};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod camera;
mod integrator;
mod ray;
mod scene;
mod tlas;
mod utility;

const ASPECT_RATIO: f32 = 1.0;
const IMAGE_WIDTH: usize = 1000;
const IMAGE_HEIGHT: usize = ((IMAGE_WIDTH as f32) / ASPECT_RATIO) as usize;
const SAMPLES_PER_PIXEL: u32 = 32;
const MAX_BOUNCES: u32 = 1024;

const ENABLE_NEE: bool = true;

fn generate_halton(base_x: u32, base_y: u32, num_samples: u32) -> Vec<glam::Vec2>
{
    (0..num_samples)
        .into_par_iter()
        .map(|i: u32| {
            // x axis
            let mut x: f32 = 0.0;
            let mut denominator_x: f32 = base_x as f32;
            let mut n_x: u32 = i;
            while n_x > 0
            {
                let multiplier: u32 = n_x % base_x;
                x += (multiplier as f32) / denominator_x;
                n_x /= base_x;
                denominator_x *= base_x as f32;
            }

            // y axis
            let mut y: f32 = 0.0;
            let mut denominator_y: f32 = base_y as f32;
            let mut n_y: u32 = i;
            while n_y > 0
            {
                let multiplier: u32 = n_y % base_y;
                y += (multiplier as f32) / denominator_y;
                n_y /= base_y;
                denominator_y *= base_y as f32;
            }

            glam::Vec2::new(x, y)
        })
        .collect()
}

fn main()
{
    println!("Loading images...");
    let env = image::io::Reader::open("images/env/cannon_4k.png").unwrap().decode();
    if env.is_err()
    {
        println!("{:?}", env);
    }

    //Materials
    println!("Creating materials...");

    let diffuse_gray = Lambertian::new(glam::Vec3A::new(0.73, 0.73, 0.73));
    let diffuse_green = Lambertian::new(glam::Vec3A::new(0.12, 0.45, 0.15));
    let diffuse_red = Lambertian::new(glam::Vec3A::new(0.65, 0.05, 0.05));
    let diffuse_blue = Lambertian::new(glam::Vec3A::new(0.05, 0.05, 0.25));
    let ggx_blue = GGX::new_metal(glam::Vec3A::new(0.1, 0.1, 0.45), 0.4);
    let brown_glass_ggx = GGX::new_dielectric(glam::Vec3A::new(0.04, 0.062, 0.07), 1.5, glam::Vec3A::splat(0.95), 0.1);
    let clear_glass_ggx = GGX::new_dielectric(glam::Vec3A::ZERO, 1.5, glam::Vec3A::ONE, 0.0);
    let glass = Dielectric::new(glam::Vec3A::splat(0.95), glam::Vec3A::new(0.04, 0.062, 0.07), 1.5);
    let mirror = Specular::new(glam::Vec3A::ONE);

    let light = Emissive::new(glam::Vec3A::splat(15.0));
    let off_light = Emissive::new(glam::Vec3A::ZERO);

    //Models and BVHs
    println!("Loading models, building BVHs...\n");

    let models: Vec<Model> = vec![
        Model::new("models/cornell/cb_light.obj", &light),
        Model::new("models/cornell/cb_main.obj", &diffuse_gray),
        Model::new("models/cornell/cb_right.obj", &diffuse_red),
        Model::new("models/cornell/cb_left.obj", &diffuse_green),
        // //Model::new("models/cornell/cb_box_tall.obj", &diffuse_gray),
        // //Model::new("models/cornell/cb_box_short.obj", &diffuse_gray),
        // //Model::new("models/sphere_offset.obj", &glass),
        Model::new("models/zenobia.obj", &ggx_blue),
        Model::new("models/cornell/dragon.obj", &glass),
        // Model::new("models/sphere.obj", &clear_glass_ggx),
    ];

    let scene: Scene = Scene::new(models);

    //Camera
    println!("Initialising camera...");
    let look_from: glam::Vec3A = glam::Vec3A::new(0.0, 50.0, 1000.0);
    let look_at: glam::Vec3A = glam::Vec3A::new(0.0, 50.0, 0.0);
    let up_vector: glam::Vec3A = glam::Vec3A::Y;

    let focal_distance: f32 = glam::Vec3A::length(look_at - look_from);

    let cam: Camera = Camera::new(look_from, look_at, up_vector, 40.0, ASPECT_RATIO, 0.0, focal_distance);

    //Render
    let sample_points: Vec<glam::Vec2> = generate_halton(2, 3, SAMPLES_PER_PIXEL);

    println!("Starting path-tracing...");
    let render_begin = std::time::Instant::now();

    let image_data: Vec<glam::Vec3A> = (0..(IMAGE_WIDTH * IMAGE_HEIGHT))
        .into_par_iter()
        .map(|i: usize| {
            let x: usize = i % IMAGE_WIDTH;
            let y: usize = IMAGE_HEIGHT - 1 - (i / IMAGE_WIDTH);

            sample_points
                .iter()
                .fold(glam::Vec3A::ZERO, |accumulated: glam::Vec3A, offset: &glam::Vec2| {
                    let u: f32 = (x as f32 + offset.x) / (IMAGE_WIDTH as f32);
                    let v: f32 = (y as f32 + offset.y) / (IMAGE_HEIGHT as f32);

                    let ray: Ray = cam.create_ray(u, v);
                    accumulated + integrator::integrate(ray, &scene, &env, MAX_BOUNCES)
                })
                / (SAMPLES_PER_PIXEL as f32)
        })
        .collect();

    println!("Finished path-tracing, took {} seconds", render_begin.elapsed().as_secs());

    println!("Converting data...");
    let data: Vec<u8> = image_data.par_iter().flat_map_iter(integrator::map_colour).collect();

    println!("Saving output...");
    image::save_buffer(
        "images/output.png",
        data.as_slice(),
        IMAGE_WIDTH as u32,
        IMAGE_HEIGHT as u32,
        image::ColorType::Rgb8,
    )
    .expect("Failed writing output image");
}

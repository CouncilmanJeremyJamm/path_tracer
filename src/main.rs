#![allow(clippy::upper_case_acronyms, clippy::new_ret_no_self)]
#![feature(allocator_api, array_chunks)]
#![feature(box_into_inner)]
extern crate core;

use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use rayon::prelude::*;

use tlas::blas::primitive::material::*;
use tlas::blas::primitive::model::{HitInfo, Model};
use tlas::blas::primitive::Triangle;

use crate::camera::Camera;
use crate::image_helper::ImageHelper;
use crate::ray::Ray;
use crate::sampling::SobolSampler;
use crate::scene::Scene;
use crate::tlas::TLAS;
use crate::utility::{EPSILON, INFINITY};
use crate::volume::Volume;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod camera;
mod image_helper;
mod integrator;
mod ray;
mod sampling;
mod scene;
mod tlas;
mod utility;

const ASPECT_RATIO: f32 = 1.0;
const IMAGE_WIDTH: usize = 1000;
const IMAGE_HEIGHT: usize = ((IMAGE_WIDTH as f32) / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: u32 = 256;
const NUM_POINTS: usize = SAMPLES_PER_PIXEL as usize * 2;
const MAX_BOUNCES: u32 = 1024;

const ENABLE_NEE: bool = true;

fn main()
{
    println!("Loading images...");
    let env = ImageHelper::load_image("images/env/cannon_4k.png");

    //Materials
    println!("Creating materials...");

    let volume = Volume::new(glam::Vec3A::new(0.4, 0.62, 0.7), 0.1, 1.0 / 200.0, 0.6);

    let diffuse_gray = Lambertian::new(glam::Vec3A::new(0.73, 0.73, 0.73));
    let diffuse_green = Lambertian::new(glam::Vec3A::new(0.12, 0.45, 0.15));
    let diffuse_red = Lambertian::new(glam::Vec3A::new(0.65, 0.05, 0.05));
    let diffuse_blue = Lambertian::new(glam::Vec3A::new(0.05, 0.05, 0.25));
    let ggx_blue = GGX::new_metal(glam::Vec3A::new(0.1, 0.1, 0.45), 0.4);
    let brown_glass_ggx = GGX::new_dielectric(glam::Vec3A::splat(0.95), 0.2, 1.5, Some(volume));
    let clear_glass_ggx = GGX::new_dielectric(glam::Vec3A::ONE, 0.0, 1.5, None);
    let glass = Dielectric::new(glam::Vec3A::splat(0.95), 1.5, Some(volume));
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
        Model::new("models/cornell/dragon.obj", &brown_glass_ggx),
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
    let sobol: SobolSampler<NUM_POINTS> = SobolSampler::generate_sobol();

    println!("Starting path-tracing...");
    let render_begin = std::time::Instant::now();

    let output_data: Vec<glam::Vec3A> = (0..(IMAGE_WIDTH * IMAGE_HEIGHT))
        .into_par_iter()
        .map(|i: usize| {
            let x: usize = i % IMAGE_WIDTH;
            let y: usize = IMAGE_HEIGHT - 1 - (i / IMAGE_WIDTH);

            let mut rng: TlsWyRand = nanorand::tls_rng();
            let mut accumulated: glam::Vec3A = glam::Vec3A::ZERO;

            let seed: u32 = rng.generate();

            for index in 0..SAMPLES_PER_PIXEL
            {
                let offset: glam::Vec2 = sobol.get_ss_sobol(index as u32, seed) - glam::Vec2::splat(0.5);

                let u: f32 = (x as f32 + offset.x) / (IMAGE_WIDTH as f32);
                let v: f32 = (y as f32 + offset.y) / (IMAGE_HEIGHT as f32);

                let ray: Ray = cam.create_ray(u, v);
                accumulated += integrator::integrate(ray, &scene, &env, &mut rng, MAX_BOUNCES);
            }

            accumulated / (SAMPLES_PER_PIXEL as f32)
        })
        .collect();

    println!("Finished path-tracing, took {} seconds", render_begin.elapsed().as_secs());

    let output_image = ImageHelper::new(output_data, glam::UVec2::new(IMAGE_WIDTH as u32, IMAGE_HEIGHT as u32));

    output_image.write_image("images/output.png").expect("Failed writing output image")
}

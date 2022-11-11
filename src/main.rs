#![allow(clippy::upper_case_acronyms, clippy::new_ret_no_self)]
#![feature(allocator_api, array_chunks)]
#![feature(box_into_inner)]
#![feature(adt_const_params)]

use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use rayon::prelude::*;
use winit::dpi::PhysicalSize;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use tlas::blas::primitive::material::*;
use tlas::blas::primitive::model::{HitInfo, Model};
use tlas::blas::primitive::Triangle;

use crate::camera::Camera;
use crate::image_helper::ImageHelper;
use crate::ray::Ray;
use crate::sampling::SobolSampler;
use crate::scene::Scene;
use crate::state::State;
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
mod state;
mod tlas;
mod utility;

const ASPECT_RATIO: f32 = 16.0 / 9.0;
const IMAGE_WIDTH: usize = 1024;
const IMAGE_HEIGHT: usize = ((IMAGE_WIDTH as f32) / ASPECT_RATIO) as usize;

const SAMPLES_PER_PIXEL: u32 = 256;
const NUM_POINTS: usize = SAMPLES_PER_PIXEL as usize * 2;
const MAX_BOUNCES: u32 = 1024;

const ENABLE_NEE: bool = true;

fn main() { pollster::block_on(run()); }

async fn run()
{
    rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get() - 1).build_global().unwrap();

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
    let glass = Dielectric::new(glam::Vec3A::splat(0.95), 1.5, None);
    let mirror = Specular::new(glam::Vec3A::ONE);

    let light = Emissive::new(glam::Vec3A::splat(15.0));

    //Models and BVHs
    println!("Loading models, building BVHs...\n");

    let rotation: glam::Quat = glam::Quat::from_rotation_y(std::f32::consts::PI);
    let translation: glam::Vec3 = glam::Vec3::new(0.0, 200.0, 0.0);

    let models: Vec<Model> = vec![
        Model::new("models/cornell/cb_light.obj", light, vec![glam::Affine3A::IDENTITY]),
        Model::new("models/cornell/cb_main.obj", diffuse_gray, vec![glam::Affine3A::IDENTITY]),
        Model::new("models/cornell/cb_right.obj", diffuse_red, vec![glam::Affine3A::IDENTITY]),
        Model::new("models/cornell/cb_left.obj", diffuse_green, vec![glam::Affine3A::IDENTITY]),
        // Model::new("models/cornell/cb_box_tall.obj", diffuse_blue, vec![glam::Affine3A::IDENTITY]),
        // Model::new("models/cornell/cb_box_short.obj", diffuse_gray, vec![glam::Affine3A::IDENTITY]),
        // Model::new("models/sphere_offset.obj", glass, vec![glam::Affine3A::IDENTITY]),
        // Model::new("models/zenobia.obj", ggx_blue, vec![glam::Affine3A::IDENTITY]),
        // Model::new("models/no_neck.obj", glass, vec![glam::Affine3A::IDENTITY]),
        Model::new(
            "models/cornell/dragon.obj",
            brown_glass_ggx,
            vec![glam::Affine3A::IDENTITY, glam::Affine3A::from_rotation_translation(rotation, translation)],
        ),
    ];

    let scene: Scene = Scene::new(models);

    //Camera
    println!("Initialising camera...");
    //let look_from: glam::Vec3A = glam::Vec3A::new(0.0, 50.0, 1000.0);
    let look_from: glam::Vec3A = glam::Vec3A::new(0.0, 50.0, 1000.0);
    let look_at: glam::Vec3A = glam::Vec3A::new(0.0, 50.0, 0.0);

    let focal_distance: f32 = glam::Vec3A::length(look_at - look_from);

    let mut cam: Camera = Camera::new(look_from, look_at, 60.0, ASPECT_RATIO, 0.0, focal_distance);
    let mut last_inv_proj: glam::Mat4 = (cam.matrix * cam.inv_projection).inverse();

    //Render
    let sobol: SobolSampler<NUM_POINTS> = SobolSampler::generate_sobol();

    println!("Starting path-tracing...");

    let mut data: Box<[glam::Vec4; IMAGE_WIDTH * IMAGE_HEIGHT]> = Box::new([glam::Vec4::ZERO; IMAGE_WIDTH * IMAGE_HEIGHT]);
    let mut position: Box<[glam::Vec4; IMAGE_WIDTH * IMAGE_HEIGHT]> = Box::new([glam::Vec4::ZERO; IMAGE_WIDTH * IMAGE_HEIGHT]);
    let mut id: Box<[u32; IMAGE_WIDTH * IMAGE_HEIGHT]> = Box::new([0; IMAGE_WIDTH * IMAGE_HEIGHT]);

    env_logger::init();
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Path Tracer")
        .with_inner_size(PhysicalSize::new(IMAGE_WIDTH as f32, IMAGE_HEIGHT as f32))
        .with_visible(true)
        .build(&event_loop)
        .unwrap();

    // State::new uses async code, so we're going to wait for it to finish
    let mut state: State = State::new(&window).await;

    let mut last_time: std::time::Instant = std::time::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let now = std::time::Instant::now();

        let dt: f32 = (now - last_time).as_secs_f32();
        last_time = now;

        if !cam.input(&event, &window.id(), dt)
        {
            match event
            {
                Event::WindowEvent { window_id, ref event } if window_id == window.id() => match event
                {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) =>
                    {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } =>
                    {
                        state.resize(**new_inner_size);
                    }
                    _ =>
                    {}
                },
                Event::RedrawRequested(window_id) if window_id == window.id() =>
                {
                    match state.render()
                    {
                        Ok(_) =>
                        {}
                        // Reconfigure the surface if it's lost or outdated
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.resize(state.size),
                        // The system is out of memory, we should probably quit
                        Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                        // We're ignoring timeouts
                        Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                    }
                }
                Event::MainEventsCleared =>
                {
                    data.par_iter_mut()
                        .zip_eq(position.par_iter_mut())
                        .zip_eq(id.par_iter_mut())
                        .map(|((c, p), id)| (c, p, id))
                        .enumerate()
                        .for_each(|(i, (c, p, id)): (usize, (&mut glam::Vec4, &mut glam::Vec4, &mut u32))| {
                            let x: usize = i % IMAGE_WIDTH;
                            let y: usize = i / IMAGE_WIDTH;

                            let mut rng: TlsWyRand = nanorand::tls_rng();

                            // TODO: new seed each time, or just 1 constant seed per entry?
                            let seed: u32 = rng.generate();
                            let offset: glam::Vec2 = sobol.get_ss_sobol(c.w as u32, seed) - 0.5;

                            let u: f32 = (x as f32 + offset.x) / (IMAGE_WIDTH as f32);
                            let v: f32 = (y as f32 + offset.y) / (IMAGE_HEIGHT as f32);

                            let ray: Ray = cam.create_ray(u, v);

                            let (colour, position, new_id): (glam::Vec4, glam::Vec4, u8) =
                                integrator::integrate(ray, &scene, &env, &mut rng, MAX_BOUNCES);

                            *c = colour;
                            *p = position;
                            *id = (*id << 16) | (new_id as u32);
                        });

                    state.update(
                        data.as_slice(),
                        position.as_slice(),
                        id.as_slice(),
                        &(cam.matrix * cam.inv_projection).inverse(),
                        &last_inv_proj,
                    );
                    last_inv_proj = (cam.matrix * cam.inv_projection).inverse();

                    window.request_redraw();
                }
                _ =>
                {}
            }
        }
    });
}

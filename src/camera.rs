use winit::event::{DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::window::WindowId;

use crate::Ray;

pub struct Camera
{
    yaw: f32,
    pitch: f32,

    pub(crate) matrix: glam::Affine3A,
    pub(crate) inv_projection: glam::Mat4,
}

impl Camera
{
    pub fn new(origin: glam::Vec3A, target: glam::Vec3A, fov: f32, aspect_ratio: f32, _: f32, _: f32) -> Self
    {
        let matrix: glam::Affine3A = glam::Affine3A::look_at_rh(origin.into(), target.into(), glam::Vec3::Y).inverse();
        let projection: glam::Mat4 = glam::Mat4::perspective_infinite_rh(fov.to_radians(), aspect_ratio, 1.0);
        let inv_projection: glam::Mat4 = projection.inverse();

        let (pitch, yaw, _): (f32, f32, f32) = matrix.to_scale_rotation_translation().1.to_euler(glam::EulerRot::YXZ);

        Self {
            yaw,
            pitch,
            matrix,
            inv_projection,
        }
    }

    fn update_origin(&mut self, dx: f32, dz: f32, dt: f32) -> bool
    {
        let sensitivity: f32 = 5.0e5;
        self.matrix.translation += self.matrix.transform_vector3a(glam::Vec3A::new(dx, 0.0, -dz)) * dt * sensitivity;

        true
    }

    fn update_rotation(&mut self, dx: f32, dy: f32, dt: f32) -> bool
    {
        let sensitivity: f32 = 1.0e4;

        self.yaw -= dy * dt * sensitivity;
        self.pitch -= dx * dt * sensitivity;
        self.matrix = glam::Affine3A::from_rotation_translation(
            glam::Quat::from_euler(glam::EulerRot::YXZ, self.pitch, self.yaw, 0.0),
            self.matrix.translation.into(),
        );

        true
    }

    pub fn input(&mut self, event: &Event<()>, target_window_id: &WindowId, dt: f32) -> bool
    {
        match event
        {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => self.update_rotation(delta.0 as f32, delta.1 as f32, dt),
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                window_id,
            } if window_id == target_window_id => match input
            {
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::W),
                    ..
                } => self.update_origin(0.0, 1.0, dt),
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::S),
                    ..
                } => self.update_origin(0.0, -1.0, dt),
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::A),
                    ..
                } => self.update_origin(-1.0, 0.0, dt),
                KeyboardInput {
                    state: ElementState::Pressed,
                    virtual_keycode: Some(VirtualKeyCode::D),
                    ..
                } => self.update_origin(1.0, 0.0, dt),
                _ => false,
            },
            _ => false,
        }
    }

    pub fn create_ray(&self, s: f32, t: f32) -> Ray
    {
        let ndc: glam::Vec3 = glam::Vec3::new(s * 2.0 - 1.0, t * 2.0 - 1.0, 0.0);

        let point: glam::Vec3A = (self.matrix * self.inv_projection).project_point3(ndc).into();
        let dir: glam::Vec3A = (point - self.matrix.translation).normalize();

        // let ndc: glam::Vec3A = glam::Vec3A::new(s * 2.0 - 1.0, t * 2.0 - 1.0, 0.0);
        // let dir: glam::Vec3A = (self.matrix * self.inv_projection).transform_point3a(ndc);

        Ray::new(self.matrix.translation, dir)
    }
}

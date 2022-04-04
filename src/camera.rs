use crate::Ray;

pub struct Camera
{
    pub origin: glam::Vec3A,
    pub ndc_to_world: glam::Mat4,
    pub camera_to_clip: glam::Mat4,
    pub world_to_camera: glam::Mat4,
}

impl Camera
{
    pub fn new(
        origin: glam::Vec3A,
        look_at: glam::Vec3A,
        up_vector: glam::Vec3A,
        fov: f32,
        aspect_ratio: f32,
        _aperture_ratio: f32,
        _focal_distance: f32,
    ) -> Self
    {
        let theta: f32 = fov.to_radians();

        let camera_to_clip: glam::Mat4 = glam::Mat4::perspective_infinite_rh(theta, aspect_ratio, 1.0);
        let world_to_camera: glam::Mat4 = glam::Mat4::look_at_rh(origin.into(), look_at.into(), up_vector.into());
        let ndc_to_world: glam::Mat4 = glam::Mat4::inverse(&(camera_to_clip * world_to_camera));

        Self {
            origin,
            ndc_to_world,
            camera_to_clip,
            world_to_camera,
        }
    }

    pub fn create_ray(&self, s: f32, t: f32) -> Ray
    {
        let point: glam::Vec4 = self.ndc_to_world * glam::Vec4::new(s * 2.0 - 1.0, t * 2.0 - 1.0, 0.0, 1.0);
        let dir: glam::Vec3A = glam::Vec3A::normalize(glam::Vec3A::from(point.truncate()) - self.origin);

        Ray::new(self.origin, dir)
    }
}

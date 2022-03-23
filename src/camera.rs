use crate::Ray;

pub struct Camera
{
    pub origin: glam::DVec3,
    pub ndc_to_world: glam::DMat4,
    pub camera_to_clip: glam::DMat4,
    pub world_to_camera: glam::DMat4,
}

impl Camera
{
    pub fn new(
        origin: glam::DVec3,
        look_at: glam::DVec3,
        up_vector: glam::DVec3,
        fov: f64,
        aspect_ratio: f64,
        _aperture_ratio: f64,
        _focal_distance: f64,
    ) -> Self
    {
        let theta: f64 = fov.to_radians();

        let camera_to_clip: glam::DMat4 =
            glam::DMat4::perspective_infinite_rh(theta, aspect_ratio, 1.0);
        let world_to_camera: glam::DMat4 = glam::DMat4::look_at_rh(origin, look_at, up_vector);
        let ndc_to_world: glam::DMat4 = glam::DMat4::inverse(&(camera_to_clip * world_to_camera));

        Self {
            origin,
            ndc_to_world,
            camera_to_clip,
            world_to_camera,
        }
    }

    pub fn create_ray(&self, s: f64, t: f64) -> Ray
    {
        let point: glam::DVec4 =
            self.ndc_to_world * glam::DVec4::new(s * 2.0 - 1.0, t * 2.0 - 1.0, 0.0, 1.0);
        let dir: glam::DVec3 = glam::DVec3::normalize(point.truncate() - self.origin);

        Ray::new(self.origin, dir)
    }
}

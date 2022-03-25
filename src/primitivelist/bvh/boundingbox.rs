use crate::INFINITY;
use crate::ray::Ray;
use crate::utility::EPSILON;

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone)]
pub struct AABB
{
    pub minimum: glam::Vec3A,
    pub maximum: glam::Vec3A,
}

impl Default for AABB
{
    fn default() -> Self
    {
        Self {
            minimum: glam::Vec3A::new(0.0, 0.0, 0.0),
            maximum: glam::Vec3A::new(0.0, 0.0, 0.0),
        }
    }
}

impl AABB
{
    pub fn new(minimum: glam::Vec3A, maximum: glam::Vec3A) -> Self
    {
        debug_assert!(minimum.le(&maximum));

        Self {
            minimum,
            maximum: glam::Vec3A::max(
                maximum,
                minimum + glam::Vec3A::new(EPSILON, EPSILON, EPSILON),
            ),
        }
    }

    pub fn identity() -> Self
    {
        Self {
            minimum: glam::Vec3A::new(INFINITY, INFINITY, INFINITY),
            maximum: glam::Vec3A::new(-INFINITY, -INFINITY, -INFINITY),
        }
    }

    pub fn surface_area(&self) -> f32
    {
        let v: glam::Vec3A = self.maximum - self.minimum;

        2.0 * (v.x * (v.y + v.z) + v.y * (v.x + v.z) + v.z * (v.y + v.x))
    }

    pub fn intersect(&self, ray: &Ray, t_max: f32) -> bool
    {
        let t0: glam::Vec3A = (self.minimum - ray.origin) * ray.inv_direction;
        let t1: glam::Vec3A = (self.maximum - ray.origin) * ray.inv_direction;

        let t_smaller: glam::Vec4 = glam::Vec3A::min(t0, t1).extend(EPSILON);
        let t_bigger: glam::Vec4 = glam::Vec3A::max(t0, t1).extend(t_max);

        //tMin < tMax
        t_smaller.max_element() < t_bigger.min_element()
    }

    pub fn intersect_t(&self, ray: &Ray, t_max: f32) -> Option<f32>
    {
        let t0: glam::Vec3A = (self.minimum - ray.origin) * ray.inv_direction;
        let t1: glam::Vec3A = (self.maximum - ray.origin) * ray.inv_direction;

        let t_smaller: glam::Vec4 = glam::Vec3A::min(t0, t1).extend(EPSILON);
        let t_bigger: glam::Vec4 = glam::Vec3A::max(t0, t1).extend(t_max);

        let t: f32 = t_smaller.max_element();
        if t < t_bigger.min_element() {
            Some(t)
        } else {
            None
        }
    }
}

pub fn surrounding_box(a: &AABB, b: &AABB) -> AABB
{
    let minimum: glam::Vec3A = glam::Vec3A::min(a.minimum, b.minimum);
    let maximum: glam::Vec3A = glam::Vec3A::max(a.maximum, b.maximum);
    AABB::new(minimum, maximum)
}

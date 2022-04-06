use crate::ray::Ray;
use crate::utility::EPSILON;
use crate::INFINITY;

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone)]
pub struct AABB
{
    minimum: glam::Vec3A,
    maximum: glam::Vec3A,
}

impl Default for AABB
{
    fn default() -> Self
    {
        Self {
            minimum: glam::Vec3A::ZERO,
            maximum: glam::Vec3A::ZERO,
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
            maximum: glam::Vec3A::max(maximum, minimum + glam::Vec3A::new(EPSILON, EPSILON, EPSILON)),
        }
    }

    pub fn identity() -> Self
    {
        Self {
            minimum: glam::Vec3A::new(INFINITY, INFINITY, INFINITY),
            maximum: glam::Vec3A::new(-INFINITY, -INFINITY, -INFINITY),
        }
    }

    pub fn compare(&self, other: &AABB, axis: u8) -> std::cmp::Ordering { self.minimum[axis as usize].total_cmp(&other.minimum[axis as usize]) }

    pub fn length(&self) -> glam::Vec3A { self.maximum - self.minimum }

    pub fn surface_area(&self) -> f32
    {
        let v: glam::Vec3A = self.maximum - self.minimum;

        2.0 * glam::Vec3A::dot(v, glam::Vec3A::new(v.y + v.z, v.x + v.z, v.x + v.y))
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
        (t < t_bigger.min_element()).then_some(t)
    }
}

pub fn surrounding_box(a: &AABB, b: &AABB) -> AABB
{
    let minimum: glam::Vec3A = glam::Vec3A::min(a.minimum, b.minimum);
    let maximum: glam::Vec3A = glam::Vec3A::max(a.maximum, b.maximum);
    AABB::new(minimum, maximum)
}

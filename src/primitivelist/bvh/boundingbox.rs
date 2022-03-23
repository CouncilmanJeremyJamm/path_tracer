use crate::ray::Ray;
use crate::utility::EPSILON;

pub struct AABB
{
    pub minimum: glam::DVec3,
    pub maximum: glam::DVec3,
}

impl Default for AABB
{
    fn default() -> Self
    {
        Self {
            minimum: glam::DVec3::new(0.0, 0.0, 0.0),
            maximum: glam::DVec3::new(0.0, 0.0, 0.0),
        }
    }
}

impl AABB
{
    pub fn new(minimum: glam::DVec3, maximum: glam::DVec3) -> Self
    {
        debug_assert!(minimum.le(&maximum));

        Self {
            minimum,
            maximum: glam::DVec3::max(
                maximum,
                minimum + glam::DVec3::new(EPSILON, EPSILON, EPSILON),
            ),
        }
    }

    pub fn intersect(&self, ray: &Ray, t_max: f64) -> bool
    {
        let t_min: f64 = EPSILON;

        let t0: glam::DVec3 = (self.minimum - ray.origin) * ray.inv_direction;
        let t1: glam::DVec3 = (self.maximum - ray.origin) * ray.inv_direction;

        let t_smaller: glam::DVec3 = glam::DVec3::min(t0, t1);
        let t_bigger: glam::DVec3 = glam::DVec3::max(t0, t1);

        //tMin < tMax
        t_min.max(t_smaller.max_element()) < t_max.min(t_bigger.min_element())
    }

    pub fn intersect_t(&self, ray: &Ray, t_max: f64) -> Option<f64>
    {
        let t_min: f64 = EPSILON;

        let t0: glam::DVec3 = (self.minimum - ray.origin) * ray.inv_direction;
        let t1: glam::DVec3 = (self.maximum - ray.origin) * ray.inv_direction;

        let t_smaller: glam::DVec3 = glam::DVec3::min(t0, t1);
        let t_bigger: glam::DVec3 = glam::DVec3::max(t0, t1);

        let t: f64 = t_min.max(t_smaller.max_element());
        if t < t_max.min(t_bigger.min_element()) {
            Some(t)
        } else {
            None
        }
    }
}

pub fn surrounding_box(a: &AABB, b: &AABB) -> AABB
{
    let minimum: glam::DVec3 = glam::DVec3::min(a.minimum, b.minimum);
    let maximum: glam::DVec3 = glam::DVec3::max(a.maximum, b.maximum);
    AABB::new(minimum, maximum)
}

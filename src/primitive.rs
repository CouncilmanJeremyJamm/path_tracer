use crate::primitive::model::{HitInfo, Vertex};
use crate::primitivelist::bvh::boundingbox::AABB;
use crate::ray::Ray;
use crate::utility::EPSILON;
use crate::Material;

pub mod model;

#[derive(Clone)]
pub struct Triangle<'a>
{
    vertices: [Vertex; 3],
    a: glam::DVec3,
    b: glam::DVec3,

    pub material: &'a (dyn Material + Sync + Send),
}

impl<'a> Triangle<'a>
{
    //TODO: https://stackoverflow.com/questions/34707994/how-to-pass-a-box-value-to-a-function
    pub fn new(
        va: Vertex,
        vb: Vertex,
        vc: Vertex,
        material: &'a (dyn Material + Sync + Send),
    ) -> Self
    {
        let a: glam::DVec3 = vb.position - va.position;
        let b: glam::DVec3 = vc.position - va.position;
        Self {
            vertices: [va, vb, vc],
            a,
            b,
            material,
        }
    }

    pub fn get_normal(&self, u: f64, v: f64) -> glam::DVec3
    {
        let w: f64 = 1.0 - u - v;
        (self.vertices[0].normal * w)
            + (self.vertices[1].normal * u)
            + (self.vertices[2].normal * v)
    }

    pub fn intersect(&self, ray: &Ray, t_max: f64) -> Option<HitInfo>
    {
        let p_vec: glam::DVec3 = glam::DVec3::cross(ray.direction, self.b);
        let determinant: f64 = glam::DVec3::dot(self.a, p_vec);

        if determinant.abs() < EPSILON {
            return None;
        }

        let inv_determinant: f64 = 1.0 / determinant;

        let t_vec: glam::DVec3 = ray.origin - self.vertices[0].position;
        let u: f64 = glam::DVec3::dot(t_vec, p_vec) * inv_determinant;

        if u < 0.0 || u > 1.0 {
            return None;
        }

        let q_vec: glam::DVec3 = glam::DVec3::cross(t_vec, self.a);
        let v: f64 = glam::DVec3::dot(ray.direction, q_vec) * inv_determinant;

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t: f64 = glam::DVec3::dot(self.b, q_vec) * inv_determinant;
        //println!("{}", t);
        if t < EPSILON || t > t_max {
            return None;
        }

        let n: glam::DVec3 = self.get_normal(u, v);
        let face_forward: bool = glam::DVec3::dot(ray.direction, n) < 0.0;

        let hit_info: HitInfo = HitInfo {
            normal: if face_forward { n } else { -n },
            local: glam::DVec2::new(u, v),
            t,
            front_facing: face_forward,
        };

        Some(hit_info)
    }

    pub fn intersect_bool(&self, ray: &Ray, t_max: f64) -> bool
    {
        let p_vec: glam::DVec3 = glam::DVec3::cross(ray.direction, self.b);
        let determinant: f64 = glam::DVec3::dot(self.a, p_vec);

        if determinant.abs() < EPSILON {
            return false;
        }

        let inv_determinant: f64 = 1.0 / determinant;

        let t_vec: glam::DVec3 = ray.origin - self.vertices[0].position;
        let u: f64 = glam::DVec3::dot(t_vec, p_vec) * inv_determinant;

        if u < 0.0 || u > 1.0 {
            return false;
        }

        let q_vec: glam::DVec3 = glam::DVec3::cross(t_vec, self.a);
        let v: f64 = glam::DVec3::dot(ray.direction, q_vec) * inv_determinant;

        if v < 0.0 || u + v > 1.0 {
            return false;
        }

        let t: f64 = glam::DVec3::dot(self.b, q_vec) * inv_determinant;

        if t < EPSILON || t > t_max {
            return false;
        }

        true
    }

    pub fn create_bounding_box(&self) -> AABB
    {
        let mut minimum: glam::DVec3 = self.vertices[0].position;
        let mut maximum: glam::DVec3 = self.vertices[0].position;

        for v in &self.vertices[1..] {
            minimum = glam::DVec3::min(minimum, v.position);
            maximum = glam::DVec3::max(maximum, v.position);
        }

        AABB::new(minimum, maximum)
    }

    pub fn local_to_world(&self, u: f64, v: f64) -> glam::DVec3
    {
        (self.vertices[0].position * (1.0 - u.sqrt()))
            + (self.vertices[1].position * u.sqrt() * (1.0 - v))
            + (self.vertices[2].position * u.sqrt() * v)
    }

    pub fn area(&self) -> f64
    {
        0.5 * glam::DVec3::length(glam::DVec3::cross(self.a, self.b))
    }
}

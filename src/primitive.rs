use crate::primitive::model::{HitInfo, Vertex};
use crate::ray::Ray;
use crate::tlas::blas::bvh::boundingbox::AABB;
use crate::utility::EPSILON;

pub mod model;

pub struct Triangle
{
    vertices: [Vertex; 3],
    a: glam::Vec3A,
    b: glam::Vec3A,
}

impl Triangle
{
    pub fn new(v: &[Vertex; 3]) -> Self
    {
        let a: glam::Vec3A = v[1].position - v[0].position;
        let b: glam::Vec3A = v[2].position - v[0].position;
        Self {
            vertices: [v[0], v[1], v[2]],
            a,
            b,
        }
    }

    pub fn get_normal(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let w: f32 = 1.0 - u - v;
        (self.vertices[0].normal * w) + (self.vertices[1].normal * u) + (self.vertices[2].normal * v)
    }

    pub fn intersect(&self, ray: &Ray, t_max: f32) -> Option<HitInfo>
    {
        let p_vec: glam::Vec3A = glam::Vec3A::cross(ray.direction, self.b);
        let determinant: f32 = glam::Vec3A::dot(self.a, p_vec);

        if determinant.abs() < EPSILON
        {
            return None;
        }

        let inv_determinant: f32 = 1.0 / determinant;

        let t_vec: glam::Vec3A = ray.origin - self.vertices[0].position;
        let u: f32 = glam::Vec3A::dot(t_vec, p_vec) * inv_determinant;

        if u < 0.0 || u > 1.0
        {
            return None;
        }

        let q_vec: glam::Vec3A = glam::Vec3A::cross(t_vec, self.a);
        let v: f32 = glam::Vec3A::dot(ray.direction, q_vec) * inv_determinant;

        if v < 0.0 || u + v > 1.0
        {
            return None;
        }

        let t: f32 = glam::Vec3A::dot(self.b, q_vec) * inv_determinant;
        //println!("{}", t);
        if t < EPSILON || t > t_max
        {
            return None;
        }

        let n: glam::Vec3A = self.get_normal(u, v);
        let face_forward: bool = glam::Vec3A::dot(ray.direction, n) < 0.0;

        let hit_info: HitInfo = HitInfo {
            normal: if face_forward { n } else { -n },
            local: glam::Vec2::new(u, v),
            t,
            front_facing: face_forward,
        };

        Some(hit_info)
    }

    pub fn intersect_bool(&self, ray: &Ray, t_max: f32) -> bool
    {
        let p_vec: glam::Vec3A = glam::Vec3A::cross(ray.direction, self.b);
        let determinant: f32 = glam::Vec3A::dot(self.a, p_vec);

        if determinant.abs() < EPSILON
        {
            return false;
        }

        let inv_determinant: f32 = 1.0 / determinant;

        let t_vec: glam::Vec3A = ray.origin - self.vertices[0].position;
        let u: f32 = glam::Vec3A::dot(t_vec, p_vec) * inv_determinant;

        if u < 0.0 || u > 1.0
        {
            return false;
        }

        let q_vec: glam::Vec3A = glam::Vec3A::cross(t_vec, self.a);
        let v: f32 = glam::Vec3A::dot(ray.direction, q_vec) * inv_determinant;

        if v < 0.0 || u + v > 1.0
        {
            return false;
        }

        let t: f32 = glam::Vec3A::dot(self.b, q_vec) * inv_determinant;

        if t < EPSILON || t > t_max
        {
            return false;
        }

        true
    }

    pub fn create_bounding_box(&self) -> AABB
    {
        let mut minimum: glam::Vec3A = self.vertices[0].position;
        let mut maximum: glam::Vec3A = self.vertices[0].position;

        for v in &self.vertices[1..]
        {
            minimum = glam::Vec3A::min(minimum, v.position);
            maximum = glam::Vec3A::max(maximum, v.position);
        }

        AABB::new(minimum, maximum)
    }

    pub fn local_to_world(&self, u: f32, v: f32) -> glam::Vec3A
    {
        (self.vertices[0].position * (1.0 - u.sqrt()))
            + (self.vertices[1].position * u.sqrt() * (1.0 - v))
            + (self.vertices[2].position * u.sqrt() * v)
    }

    pub fn area(&self) -> f32 { 0.5 * glam::Vec3A::length(glam::Vec3A::cross(self.a, self.b)) }
}

use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::ray::Ray;
use crate::tlas::blas::blas_bvh::boundingbox::AABB;
use crate::tlas::blas::Vertex;
use crate::utility::EPSILON;
use crate::HitInfo;

pub mod material;
pub mod model;

pub struct Triangle
{
    positions: glam::Mat3A,
    normals: glam::Mat3A,
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
            positions: glam::Mat3A::from_cols(v[0].position, v[1].position, v[2].position),
            normals: glam::Mat3A::from_cols(v[0].normal, v[1].normal, v[2].normal),
            a,
            b,
        }
    }

    pub fn get_normal(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let w: f32 = 1.0 - u - v;
        self.normals * glam::Vec3A::new(w, u, v)
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

        let t_vec: glam::Vec3A = ray.origin - self.positions.col(0);
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

        let t_vec: glam::Vec3A = ray.origin - self.positions.col(0);
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
        let minimum: glam::Vec3A = self.positions.col(0).min(self.positions.col(1)).min(self.positions.col(2));
        let maximum: glam::Vec3A = self.positions.col(0).max(self.positions.col(1)).max(self.positions.col(2));

        AABB::new(minimum, maximum)
    }

    pub fn get_position(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let w: f32 = 1.0 - u - v;
        self.positions * glam::Vec3A::new(w, u, v)
    }

    //Obtains a random point on the primitive, distributed uniformly on the surface
    //Return the point, and the normal at that point
    pub fn random_point(&self, rng: &mut TlsWyRand) -> (glam::Vec3A, glam::Vec3A)
    {
        //Flipping sampling method from NVIDIA's RTG-1, pg. 236
        //Cannot use low-discrepancy sequences, no guarantee on distribution after folding
        let mut u: f32 = rng.generate();
        let mut v: f32 = rng.generate();

        if u + v > 1.0
        {
            u = 1.0 - u;
            v = 1.0 - v;
        }

        (self.get_position(u, v), self.get_normal(u, v))
    }

    pub fn area(&self) -> f32 { 0.5 * glam::Vec3A::length(glam::Vec3A::cross(self.a, self.b)) }
}

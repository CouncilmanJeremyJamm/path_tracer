use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::ray::Ray;
use crate::tlas::blas::blas_bvh::boundingbox::AABB;
use crate::tlas::blas::Vertex;
use crate::utility::EPSILON;
use crate::HitInfo;

pub mod material;
pub mod model;

/// Triangles are the only primitive implemented in the path tracer
pub struct Triangle
{
    /// Positions of vertices A, B, and C
    positions: glam::Mat3A,
    /// Normals at vertices A, B, and C
    normals: glam::Mat3A,
    // Precomputed vectors to use in intersection tests
    /// `n`, with `d` packed in the `w`-component
    n0: glam::Vec4,
    /// `n1`, with `d1` packed in the `w`-component
    n1: glam::Vec4,
    /// `n2`, with `d2` packed in the `w`-component
    n2: glam::Vec4,
}

impl Triangle
{
    pub fn new(v: &[Vertex; 3]) -> Self
    {
        // Precompute all possible values used in intersection tests
        let ab: glam::Vec3A = v[1].position - v[0].position;
        let ac: glam::Vec3A = v[2].position - v[0].position;

        let n0: glam::Vec3A = glam::Vec3A::cross(ab, ac);
        let d0: f32 = glam::Vec3A::dot(n0, v[0].position);
        let scale: f32 = n0.length_squared();

        let n1: glam::Vec3A = glam::Vec3A::cross(ac, n0) / scale;
        let d1: f32 = -glam::Vec3A::dot(n1, v[0].position);

        let n2: glam::Vec3A = glam::Vec3A::cross(n0, ab) / scale;
        let d2: f32 = -glam::Vec3A::dot(n2, v[0].position);

        Self {
            positions: glam::Mat3A::from_cols(v[0].position, v[1].position, v[2].position),
            normals: glam::Mat3A::from_cols(v[0].normal, v[1].normal, v[2].normal),
            n0: n0.extend(d0),
            n1: n1.extend(d1),
            n2: n2.extend(d2),
        }
    }

    /// Computes the normal at the barycentric coordinates (u, v)
    pub fn get_normal(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let w: f32 = 1.0 - u - v;
        self.normals * glam::Vec3A::new(w, u, v)
    }

    /// Computes the world position at the barycentric coordinates (u, v)
    pub fn get_position(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let w: f32 = 1.0 - u - v;
        self.positions * glam::Vec3A::new(w, u, v)
    }

    /// Obtains a random point on the primitive, distributed uniformly on the surface
    /// # Returns
    /// (`p`, `n`)
    /// * `p` - world position of the sampled point
    /// * `n` - normal at the sampled point
    pub fn random_point(&self, rng: &mut TlsWyRand) -> (glam::Vec3A, glam::Vec3A)
    {
        //Diagonal flip sampling method from NVIDIA's RTG-1, pg. 236
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

    /// Computes the area of the current triangle
    pub fn area(&self) -> f32 { 0.5 * glam::Vec3A::length(glam::Vec3A::from(self.n0)) }

    /// Returns the AABB of the current triangle
    pub fn create_bounding_box(&self) -> AABB
    {
        let minimum: glam::Vec3A = self.positions.col(0).min(self.positions.col(1)).min(self.positions.col(2));
        let maximum: glam::Vec3A = self.positions.col(0).max(self.positions.col(1)).max(self.positions.col(2));

        AABB::new(minimum, maximum)
    }

    /// Implementation of *Yet Faster Ray-Triangle Intersection* by Havel & Herout
    ///
    /// This is labelled as *naive* as the ray origin is not translated before testing, reducing precision
    /// # Returns
    /// `Some(a)` or `None`
    /// * `a` - glam::Vec4 containing the components:
    ///     * `x`: `t * det`
    ///     * `y`: `u * det`
    ///     * `z`: `v * det`
    ///     * `w`: `det`
    ///
    /// This defers the division to obtain t, u, and v, which are not needed in the any-intersection test
    fn intersect_naive(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<glam::Vec4>
    {
        let det: f32 = glam::Vec3A::dot(ray.direction, self.n0.into());
        let td: f32 = -glam::Vec4::dot(ray.origin.extend(-1.0), self.n0);

        if (td - det * t_min).signum() != (det * t_max - td).signum()
        {
            return None;
        }

        let p: glam::Vec4 = (det * ray.origin + td * ray.direction).extend(det);

        let ud: f32 = glam::Vec4::dot(p, self.n1);

        if ud.signum() != (det - ud).signum()
        {
            return None;
        }

        let vd: f32 = glam::Vec4::dot(p, self.n2);

        if vd.signum() != (det - ud - vd).signum()
        {
            return None;
        }

        Some(glam::Vec4::new(td, ud, vd, det))
    }

    /// Full intersection test, with ray pre-translation
    pub fn intersect(&self, ray: &Ray, t_max: f32, t_estimate: f32) -> Option<HitInfo>
    {
        // Translate the ray before running the naive test
        let moved_ray: Ray = Ray {
            origin: ray.at(t_estimate),
            ..*ray
        };

        if let Some(tuvd) = self.intersect_naive(&moved_ray, EPSILON - t_estimate, t_max - t_estimate)
        {
            // Divide the returned components by det
            let [t, u, v]: [f32; 3] = (glam::Vec3A::from(tuvd) / tuvd.w).into();

            // Obtain the shading normal, and record whether the ray was incident on the inside or outside of the triangle
            let n: glam::Vec3A = self.get_normal(u, v);
            let face_forward: bool = glam::Vec3A::dot(ray.direction, n) < 0.0;

            let hit_info: HitInfo = HitInfo {
                normal: if face_forward { n } else { -n },
                local: glam::Vec2::new(u, v),
                // Adjust the t value to account for pre-translation
                t: t + t_estimate,
                front_facing: face_forward,
            };

            Some(hit_info)
        }
        else
        {
            None
        }
    }

    pub fn intersect_bool(&self, ray: &Ray, t_max: f32, t_estimate: f32) -> bool
    {
        let moved_ray: Ray = Ray {
            origin: ray.at(t_estimate),
            ..*ray
        };

        self.intersect_naive(&moved_ray, EPSILON - t_estimate, t_max - t_estimate).is_some()
    }
}

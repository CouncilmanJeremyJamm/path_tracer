use std::path::Path;

use crate::material;

pub struct HitInfo
{
    pub normal: glam::Vec3A,
    pub local: glam::Vec2,
    pub t: f32,
    pub front_facing: bool,
}

#[derive(Copy, Clone)]
pub struct Vertex
{
    pub position: glam::Vec3A,
    pub normal: glam::Vec3A,
}

pub struct VertexRef
{
    pub vertex: usize,
    pub normal: usize,
}

pub struct Model<'c>
{
    pub file_path: &'c Path,
    pub material: &'c (dyn material::Material),
}

impl<'c> Model<'c>
{
    pub fn new<P: ?Sized + AsRef<Path>>(file_path: &'c P, material: &'c (dyn material::Material)) -> Self
    {
        Self {
            file_path: file_path.as_ref(),
            material,
        }
    }
}

use std::path::Path;

use crate::Material;

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
    pub material: &'c Material,
}

impl<'c> Model<'c>
{
    pub fn new<P>(file_path: &'c P, material: &'c Material) -> Self
    where
        P: ?Sized + AsRef<Path>,
    {
        Self {
            file_path: file_path.as_ref(),
            material,
        }
    }
}

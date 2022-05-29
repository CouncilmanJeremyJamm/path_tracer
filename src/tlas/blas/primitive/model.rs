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

#[derive(Clone)]
pub struct Model<'c>
{
    pub file_path: &'c Path,
    pub material: &'c Material,
    pub matrices: Vec<glam::Affine3A>,
}

impl<'c> Model<'c>
{
    pub fn new<P>(file_path: &'c P, material: &'c Material, matrices: Vec<glam::Affine3A>) -> Self
    where
        P: ?Sized + AsRef<Path>,
    {
        for matrix in &matrices
        {
            let (scale, _, _) = matrix.to_scale_rotation_translation();
            assert_eq!(scale, glam::Vec3::ONE, "Model matrix can only contain translation and rotation");
        }

        Self {
            file_path: file_path.as_ref(),
            material,
            matrices,
        }
    }
}

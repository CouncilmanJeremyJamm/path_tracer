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
    pub file_path: &'c std::path::Path,
    pub material: &'c (dyn material::Material + Sync + Send),
}

impl<'c> Model<'c>
{
    pub fn new(file_path: &'c str, material: &'c (dyn material::Material + Sync + Send)) -> Self
    {
        Self {
            file_path: std::path::Path::new(file_path),
            material,
        }
    }
}

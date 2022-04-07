use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use bumpalo::Bump;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use primitive::model::{Model, Vertex, VertexRef};
use primitive::Triangle;

use crate::tlas::blas::blas_bvh::{BLASNode, BLASNodeType, PrimitiveInfo};
use crate::{HitInfo, Material, Ray};

pub mod blas_bvh;
pub mod primitive;

pub fn load_obj(path: &std::path::Path) -> Vec<Vertex>
{
    let mut normals: Vec<glam::Vec3A> = vec![glam::Vec3A::ZERO];
    let mut positions: Vec<glam::Vec3A> = vec![glam::Vec3A::ZERO];

    let mut vertices: Vec<Vertex> = Vec::new();

    let file: File = File::open(path).unwrap();
    let lines = BufReader::new(file).lines().filter_map(|l| l.ok());

    for line in lines
    {
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match tokens[0]
        {
            "v" =>
            //Vertex
            {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                positions.push(glam::Vec3A::new(x, y, z));
            }
            "vn" =>
            //Normal
            {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                normals.push(glam::Vec3A::new(x, y, z));
            }
            "f" =>
            //Polygon
            {
                let mut refs: Vec<VertexRef> = Vec::new();

                for token in &tokens[1..]
                {
                    // v, vt, vn
                    let indices: Vec<&str> = token.split('/').collect();

                    let v: usize = indices[0]
                        .parse::<usize>()
                        .unwrap_or_else(|_| positions.len() + indices[0].parse::<isize>().unwrap() as usize);

                    //TODO: vt, texture vertices
                    let vn: usize = indices[2]
                        .parse::<usize>()
                        .unwrap_or_else(|_| normals.len() + indices[2].parse::<isize>().unwrap() as usize);

                    refs.push(VertexRef { vertex: v, normal: vn });
                }

                for i in 1..(refs.len() - 1)
                {
                    //Triangulate
                    let p0 = &refs[0];
                    let p1 = &refs[i];
                    let p2 = &refs[i + 1];

                    for vr in [p0, p1, p2]
                    {
                        let position: glam::Vec3A = positions[vr.vertex];
                        let normal: glam::Vec3A = if vr.normal != 0
                        {
                            normals[vr.normal]
                        }
                        else
                        {
                            let u: glam::Vec3A = positions[p1.vertex] - positions[p0.vertex];
                            let v: glam::Vec3A = positions[p2.vertex] - positions[p0.vertex];
                            glam::Vec3A::cross(u, v)
                        };
                        vertices.push(Vertex { position, normal });
                    }
                }
            }
            _ =>
            //Comments, textures
            {
                continue;
            }
        }
    }

    vertices
}

pub(super) struct BLAS<'a>
{
    pub primitives: Vec<Triangle>,
    pub material: &'a (dyn Material),
    pub bvh: BLASNode,
}

impl<'a> BLAS<'a>
{
    pub fn new(model: &Model<'a>) -> Self
    {
        let timer: Instant = Instant::now();

        let vertices: Vec<Vertex> = load_obj(model.file_path);
        let primitives: Vec<Triangle> = vertices.array_chunks::<3>().map(Triangle::new).collect();

        let mut primitive_info: Vec<PrimitiveInfo> = primitives
            .par_iter()
            .enumerate()
            .map(|tuple: (usize, &Triangle)| -> PrimitiveInfo { PrimitiveInfo::new(tuple.1.create_bounding_box(), tuple.0 as u32) })
            .collect();

        let bvh: BLASNode = BLASNode::generate_blas(primitive_info.as_mut_slice(), 4);

        println!("BLAS - {:?}: \t{:?}", model.file_path.file_name().unwrap(), timer.elapsed());

        Self {
            primitives,
            material: model.material,
            bvh,
        }
    }

    pub fn intersect(&self, bump: &Bump, r: &Ray, mut t_max: f32) -> Option<(HitInfo, &Triangle, &(dyn Material))>
    {
        let mut stack: Vec<(&BLASNode, f32), _> = Vec::with_capacity_in(1, bump);
        stack.push((&self.bvh, 0.0));
        let mut closest: Option<(HitInfo, &Triangle)> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &current.node_type
            {
                BLASNodeType::Branch { left, right } =>
                {
                    let intersect_left = left.bounding_box.intersect_t(r, t_max);
                    let intersect_right = right.bounding_box.intersect_t(r, t_max);

                    if let (Some(t_enter_left), Some(t_enter_right)) = (intersect_left, intersect_right)
                    {
                        if t_enter_left < t_enter_right
                        {
                            stack.push((right, t_enter_right));
                            stack.push((left, t_enter_left));
                        }
                        else
                        {
                            stack.push((left, t_enter_left));
                            stack.push((right, t_enter_right));
                        }
                    }
                    else if let Some(t_enter_left) = intersect_left
                    {
                        stack.push((left, t_enter_left));
                    }
                    else if let Some(t_enter_right) = intersect_right
                    {
                        stack.push((right, t_enter_right));
                    }
                }
                BLASNodeType::Leaf { primitive_indices } if primitive_indices.len() == 1 =>
                //Fast path for single child
                {
                    let primitive: &Triangle = &self.primitives[primitive_indices[0] as usize];
                    if let Some(intersection) = primitive.intersect(r, t_max)
                    {
                        t_max = intersection.t;
                        closest = Some((intersection, primitive));
                    }
                }
                BLASNodeType::Leaf { primitive_indices } =>
                {
                    if let Some(intersection) = primitive_indices
                        .iter()
                        .filter_map(|i| self.primitives[*i as usize].intersect(r, t_max).zip(Some(*i)))
                        .min_by(|a, b| a.0.t.total_cmp(&b.0.t))
                    {
                        t_max = intersection.0.t;
                        closest = Some((intersection.0, &self.primitives[intersection.1 as usize]));
                    }
                }
            }
        }

        closest.map(|(intersection, primitive)| (intersection, primitive, self.material))
    }
    pub fn any_intersect(&self, bump: &Bump, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<&BLASNode, _> = Vec::with_capacity_in(1, bump);
        stack.push(&self.bvh);

        while let Some(current) = stack.pop()
        {
            if !current.bounding_box.intersect(r, t_max)
            {
                continue;
            }

            match &current.node_type
            {
                BLASNodeType::Branch { left, right } =>
                {
                    stack.push(left);
                    stack.push(right);
                }
                BLASNodeType::Leaf { primitive_indices } if primitive_indices.len() == 1 =>
                //Fast path for single child
                {
                    if self.primitives[primitive_indices[0] as usize].intersect_bool(r, t_max)
                    {
                        return true;
                    }
                }
                BLASNodeType::Leaf { primitive_indices } =>
                {
                    if primitive_indices.iter().any(|i| self.primitives[*i as usize].intersect_bool(r, t_max))
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

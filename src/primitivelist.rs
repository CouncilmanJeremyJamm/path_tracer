use std::fs::File;
use std::io::{BufRead, BufReader};

use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::primitive::model::{Model, Vertex, VertexRef};
use crate::primitive::Triangle;
use crate::primitivelist::bvh::boundingbox::AABB;
use crate::primitivelist::bvh::{BVHNode, NodeType, PrimitiveInfo};
use crate::{HitInfo, Ray};

pub(crate) mod bvh;

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

pub struct PrimitiveList<'a>
{
    pub objects: Vec<Triangle<'a>>,
    bvh: BVHNode,
}

impl<'a> PrimitiveList<'a>
{
    pub fn new() -> Self
    {
        Self {
            objects: vec![],
            bvh: BVHNode {
                bounding_box: AABB::default(),
                node_type: NodeType::Leaf {
                    primitive_indices: vec![],
                    num_objects: 0,
                },
            },
        }
    }

    pub fn copy(pl: &'a PrimitiveList) -> Self
    {
        Self {
            objects: pl.objects.clone(),
            bvh: pl.bvh.clone(),
        }
    }

    fn add_model(&mut self, model: Model<'a>)
    {
        let vertices: Vec<Vertex> = load_obj(model.file_path);

        self.objects
            .extend(vertices.array_chunks::<3>().map(|v| Triangle::new(v, model.material)));
    }

    pub fn add_models(&mut self, models: Vec<Model<'a>>)
    {
        for model in models
        {
            self.add_model(model);
        }

        let mut object_info: Vec<PrimitiveInfo> = self
            .objects
            .par_iter()
            .enumerate()
            .map(|tuple: (usize, &Triangle)| -> PrimitiveInfo { PrimitiveInfo::new(tuple.1.create_bounding_box(), tuple.0 as u32) })
            .collect();

        let begin = std::time::Instant::now();
        self.bvh = BVHNode::generate_bvh(object_info.as_mut_slice(), 4);
        println!("Finished building a BVH, took {} ms", begin.elapsed().as_millis());
    }

    pub fn intersect(&self, r: &Ray, mut t_max: f32) -> Option<(HitInfo, &Triangle)>
    {
        if !self.bvh.bounding_box.intersect(r, t_max)
        {
            return None;
        }

        let mut stack: Vec<(&BVHNode, f32)> = vec![(&self.bvh, 0.0)];
        let mut closest: Option<(HitInfo, &Triangle)> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &current.node_type
            {
                NodeType::Branch { left, right } =>
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
                NodeType::Leaf {
                    primitive_indices,
                    num_objects,
                } if *num_objects == 1 =>
                //Fast path for single child
                {
                    let primitive: &Triangle = &self.objects[primitive_indices[0] as usize];
                    if let Some(intersection) = primitive.intersect(r, t_max)
                    {
                        t_max = intersection.t;
                        closest = Some((intersection, primitive));
                    }
                }
                NodeType::Leaf {
                    primitive_indices,
                    num_objects: _,
                } =>
                {
                    if let Some(intersection) = primitive_indices
                        .iter()
                        .filter_map(|i| self.objects[*i as usize].intersect(r, t_max).zip(Some(*i)))
                        .min_by(|a, b| a.0.t.total_cmp(&b.0.t))
                    {
                        t_max = intersection.0.t;
                        closest = Some((intersection.0, &self.objects[intersection.1 as usize]));
                    }
                }
            }
        }

        closest
    }
    pub fn any_intersect(&self, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<&BVHNode> = vec![&self.bvh];

        while let Some(current) = stack.pop()
        {
            if !current.bounding_box.intersect(r, t_max)
            {
                continue;
            }

            match &current.node_type
            {
                NodeType::Branch { left, right } =>
                {
                    stack.push(left);
                    stack.push(right);
                }
                NodeType::Leaf {
                    primitive_indices,
                    num_objects,
                } if *num_objects == 1 =>
                //Fast path for single child
                {
                    if self.objects[primitive_indices[0] as usize].intersect_bool(r, t_max)
                    {
                        return true;
                    }
                }
                NodeType::Leaf {
                    primitive_indices,
                    num_objects: _,
                } =>
                {
                    if primitive_indices.iter().any(|i| self.objects[*i as usize].intersect_bool(r, t_max))
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

use std::io::BufRead;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

use crate::primitive::model::{Model, Vertex, VertexRef};
use crate::primitive::Triangle;
use crate::primitivelist::bvh::boundingbox::AABB;
use crate::primitivelist::bvh::{BVHNode, NodeType, PrimitiveInfo};
use crate::{HitInfo, Ray};

pub(crate) mod bvh;

pub fn load_obj(path: &std::path::Path) -> Vec<Vertex>
{
    let mut normals: Vec<glam::Vec3A> = vec![glam::Vec3A::new(0.0, 0.0, 0.0)];

    let mut positions: Vec<glam::Vec3A> = vec![glam::Vec3A::new(0.0, 0.0, 0.0)];

    let mut verts: Vec<Vertex> = Vec::new();

    let file = std::fs::File::open(path)
        .expect(&*("Could not open: ".to_owned() + path.to_str().unwrap()));
    let reader = std::io::BufReader::new(file);

    for (_, l) in reader.lines().enumerate() {
        let line = l.unwrap();
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens[0] == "v" {
            //Vertex
            let mut p: glam::Vec3A = glam::Vec3A::new(0.0, 0.0, 0.0);
            p.x = tokens[1].parse().unwrap();
            p.y = tokens[2].parse().unwrap();
            p.z = tokens[3].parse().unwrap();

            positions.push(p);
        } else if tokens[0] == "vn" {
            //Normal
            let mut n: glam::Vec3A = glam::Vec3A::new(0.0, 0.0, 0.0);
            n.x = tokens[1].parse().unwrap();
            n.y = tokens[2].parse().unwrap();
            n.z = tokens[3].parse().unwrap();

            normals.push(n);
        } else if tokens[0] == "f" {
            //Polygon
            let mut refs: Vec<VertexRef> = Vec::new();

            for token in &tokens[1..] {
                // v, vt, vn
                let indices: Vec<&str> = token.split('/').collect();

                let v: usize = if indices[0].parse::<isize>().unwrap() >= 0 {
                    indices[0].parse::<usize>().unwrap()
                } else {
                    positions.len() + indices[0].parse::<isize>().unwrap() as usize
                };
                //vt
                let vn: usize = if indices[2].parse::<isize>().unwrap() >= 0 {
                    indices[2].parse::<usize>().unwrap()
                } else {
                    normals.len() + indices[2].parse::<isize>().unwrap() as usize
                };

                refs.push(VertexRef {
                    vertex: v,
                    normal: vn,
                });
            }

            for i in 1..(refs.len() - 1) {
                //Triangulate
                let p0 = &refs[0];
                let p1 = &refs[i];
                let p2 = &refs[i + 1];

                for vr in [&refs[0], &refs[i], &refs[i + 1]] {
                    let position: glam::Vec3A = positions[vr.vertex];
                    let normal: glam::Vec3A = if vr.normal != 0 {
                        normals[vr.normal]
                    } else {
                        let u: glam::Vec3A = positions[p1.vertex] - positions[p0.vertex];
                        let v: glam::Vec3A = positions[p2.vertex] - positions[p0.vertex];
                        glam::Vec3A::cross(u, v)
                    };
                    verts.push(Vertex { position, normal });
                }
            }
        } else {
            //Comments, textures
            continue;
        }
    }

    // println!("load_obj: {:?}", positions);
    verts
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
                node_type: NodeType::LeafSingle { primitive_index: 0 },
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

        vertices.chunks(3).for_each(|v| {
            let va: Vertex = v[0];
            let vb: Vertex = v[1];
            let vc: Vertex = v[2];

            self.objects.push(Triangle::new(va, vb, vc, model.material));
        });
    }

    pub fn add_models(&mut self, models: Vec<Model<'a>>)
    {
        for model in models {
            self.add_model(model);
        }

        let mut object_info: Vec<PrimitiveInfo> = self
            .objects
            .par_iter()
            .enumerate()
            .map(|tuple: (usize, &Triangle)| -> PrimitiveInfo {
                PrimitiveInfo::new(tuple.1.create_bounding_box(), tuple.0 as u32)
            })
            .collect();

        let begin = std::time::Instant::now();
        self.bvh = BVHNode::generate_bvh(object_info.as_mut_slice(), 4);
        println!(
            "Finished building a BVH, took {} ms",
            begin.elapsed().as_millis()
        );
    }

    pub fn intersect(&self, r: &Ray, t_max: f32) -> Option<(HitInfo, &Triangle)>
    {
        if !self.bvh.bounding_box.intersect(r, t_max) {
            return None;
        }

        let mut stack: Vec<(&BVHNode, f32)> = vec![(&self.bvh, 0.0)];

        let mut t_max: f32 = t_max;
        let mut closest: Option<(HitInfo, &Triangle)> = None;

        while !stack.is_empty() {
            let (current, t_enter): (&BVHNode, f32) = stack.pop().unwrap();

            if t_enter > t_max {
                continue;
            }

            match &current.node_type {
                NodeType::LeafSingle { primitive_index } => {
                    let intersection: Option<HitInfo> =
                        self.objects[*primitive_index as usize].intersect(r, t_max);
                    if intersection.is_some() {
                        t_max = intersection.as_ref().unwrap().t;
                        closest = intersection.zip(Some(&self.objects[*primitive_index as usize]));
                    }
                }
                NodeType::Branch {
                    left,
                    right,
                    split_axis,
                } => {
                    let (a, b): (&BVHNode, &BVHNode) = if r.direction[*split_axis as usize] > 0.0 {
                        (left, right)
                    } else {
                        (right, left)
                    };

                    if let Some(t_enter_b) = b.bounding_box.intersect_t(r, t_max) {
                        stack.push((b, t_enter_b))
                    };
                    if let Some(t_enter_a) = a.bounding_box.intersect_t(r, t_max) {
                        stack.push((a, t_enter_a))
                    };
                }
                NodeType::LeafMultiple { primitive_indices } => {
                    if let Some(intersection) = primitive_indices
                        .iter()
                        .filter_map(|i| self.objects[*i as usize].intersect(r, t_max).zip(Some(i)))
                        .min_by(|a, b| a.0.t.total_cmp(&b.0.t))
                    {
                        t_max = intersection.0.t;
                        closest = Some((intersection.0, &self.objects[*intersection.1 as usize]));
                    }
                }
            }
        }

        closest
    }
    pub fn any_intersect(&self, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<&BVHNode> = vec![&self.bvh];

        while !stack.is_empty() {
            let current: &BVHNode = stack.pop().unwrap();

            if !current.bounding_box.intersect(r, t_max) {
                continue;
            }

            match &current.node_type {
                NodeType::LeafSingle { primitive_index } => {
                    if self.objects[*primitive_index as usize].intersect_bool(r, t_max) {
                        return true;
                    }
                }
                NodeType::Branch {
                    left,
                    right,
                    split_axis: _,
                } => {
                    stack.push(left);
                    stack.push(right);
                }
                NodeType::LeafMultiple { primitive_indices } => {
                    if primitive_indices
                        .into_par_iter()
                        .any(|i| self.objects[*i as usize].intersect_bool(r, t_max))
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

use std::cmp::Ordering;
use std::io::BufRead;

use rayon::prelude::ParallelSliceMut;

use crate::primitive::model::{Model, Vertex, VertexRef};
use crate::primitive::Triangle;
use crate::primitivelist::bvh::boundingbox::{surrounding_box, AABB};
use crate::primitivelist::bvh::{BVHNode, NodeType};
use crate::{HitInfo, Ray};

pub(crate) mod bvh;

pub fn load_obj(path: &std::path::Path) -> Vec<Vertex>
{
    let mut normals: Vec<glam::DVec3> = Vec::new();
    normals.push(glam::DVec3::new(0.0, 0.0, 0.0));

    let mut positions: Vec<glam::DVec3> = Vec::new();
    positions.push(glam::DVec3::new(0.0, 0.0, 0.0));

    let mut verts: Vec<Vertex> = Vec::new();

    let file = std::fs::File::open(path)
        .expect(&*("Could not open: ".to_owned() + path.to_str().unwrap()));
    let reader = std::io::BufReader::new(file);

    for (_, l) in reader.lines().enumerate() {
        let line = l.unwrap();
        let tokens: Vec<&str> = line.split_whitespace().collect();

        if tokens[0] == "v" {
            //Vertex
            let mut p: glam::DVec3 = glam::DVec3::new(0.0, 0.0, 0.0);
            p.x = tokens[1].parse().unwrap();
            p.y = tokens[2].parse().unwrap();
            p.z = tokens[3].parse().unwrap();

            positions.push(p);
        } else if tokens[0] == "vn" {
            //Normal
            let mut n: glam::DVec3 = glam::DVec3::new(0.0, 0.0, 0.0);
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

                let u: glam::DVec3 = positions[p1.vertex] - positions[p0.vertex];
                let v: glam::DVec3 = positions[p2.vertex] - positions[p0.vertex];
                let n: glam::DVec3 = glam::DVec3::cross(u, v);

                verts.push(Vertex {
                    position: positions[p0.vertex],
                    normal: if p0.normal != 0 {
                        normals[p0.normal]
                    } else {
                        n
                    },
                });
                verts.push(Vertex {
                    position: positions[p1.vertex],
                    normal: if p1.normal != 0 {
                        normals[p1.normal]
                    } else {
                        n
                    },
                });
                verts.push(Vertex {
                    position: positions[p2.vertex],
                    normal: if p2.normal != 0 {
                        normals[p2.normal]
                    } else {
                        n
                    },
                });
            }
        } else {
            //Comments, textures
            continue;
        }
    }

    // println!("load_obj: {:?}", positions);
    verts
}

fn create_bounding_box(indices: &[u32], objects: &[Triangle]) -> AABB
{
    let mut output_box: AABB = objects[indices[0] as usize].create_bounding_box();

    for i in &indices[1..] {
        output_box = surrounding_box(&output_box, &objects[*i as usize].create_bounding_box());
    }

    output_box
}

fn box_compare(objects: &[Triangle], a: u32, b: u32, axis: u8) -> Ordering
{
    let box_a: AABB = objects[a as usize].create_bounding_box();
    let box_b: AABB = objects[b as usize].create_bounding_box();

    box_a.minimum[axis as usize]
        .partial_cmp(&box_b.minimum[axis as usize])
        .unwrap()
}

fn generate_bvh<'a>(indices: &mut [u32], objects: &[Triangle]) -> BVHNode
{
    let object_span: usize = indices.len();

    if object_span == 1 {
        BVHNode {
            bounding_box: objects[indices[0] as usize].create_bounding_box(),
            node_type: NodeType::Leaf {
                primitive_index: indices[0],
            },
        }
    } else {
        let object_box: AABB = create_bounding_box(&indices, objects);

        //Find longest axis
        let box_length: glam::DVec3 = object_box.maximum - object_box.minimum;
        let max_length: f64 = box_length.max_element();

        let split_axis: u8 = if box_length.x == max_length {
            0u8
        } else if box_length.y == max_length {
            1u8
        } else {
            2u8
        };

        //Split along longest axis
        //TODO: don't sort twice if not needed
        let comparator =
            |a: &u32, b: &u32| -> Ordering { box_compare(objects, *a, *b, split_axis) };
        indices.par_sort_unstable_by(comparator);

        let mid: usize = object_span / 2;
        let (left_indices, right_indices): (&mut [u32], &mut [u32]) = indices.split_at_mut(mid);

        let (left, right): (Box<BVHNode>, Box<BVHNode>) = rayon::join(
            || Box::new(generate_bvh(left_indices, objects)),
            || Box::new(generate_bvh(right_indices, objects)),
        );

        BVHNode {
            bounding_box: surrounding_box(&left.bounding_box, &right.bounding_box),
            node_type: NodeType::Branch {
                split_axis,
                left,
                right,
            },
        }
    }
}

#[derive(Default)]
pub struct PrimitiveList<'a>
{
    pub objects: Vec<Triangle<'a>>,
    bvh: Box<BVHNode>,
}

impl<'a> PrimitiveList<'a>
{
    pub fn copy(pl: &'a PrimitiveList) -> Self
    {
        Self {
            objects: pl.objects.clone(),
            bvh: Box::new(BVHNode::default()),
        }
    }

    fn add_model(&mut self, model: Model<'a>)
    {
        let vertices: Vec<Vertex> = load_obj(model.file_path);

        for i in (0..vertices.len()).step_by(3) {
            let va: Vertex = vertices[i + 0];
            let vb: Vertex = vertices[i + 1];
            let vc: Vertex = vertices[i + 2];

            self.objects.push(Triangle::new(va, vb, vc, model.material));
        }
    }

    pub fn add_models(&mut self, models: Vec<Model<'a>>)
    {
        for model in models {
            self.add_model(model);
        }

        let mut indices = (0..self.objects.len() as u32).collect::<Vec<u32>>();
        let begin = std::time::Instant::now();
        self.bvh = Box::new(generate_bvh(indices.as_mut_slice(), &self.objects));
        println!(
            "Finished building a BVH, took {} ms",
            begin.elapsed().as_millis()
        );
    }

    pub fn intersect(&self, r: &Ray, t_max: f64) -> Option<(HitInfo, &Triangle)>
    {
        if !self.bvh.bounding_box.intersect(r, t_max) {
            return None;
        }

        let mut stack: Vec<(&BVHNode, f64)> = vec![(&self.bvh, 0.0)];

        let mut t_max: f64 = t_max;
        let mut closest: Option<(HitInfo, &Triangle)> = None;

        while !stack.is_empty() {
            let (current, t_enter): (&BVHNode, f64) = stack.pop().unwrap();

            if t_enter > t_max {
                continue;
            }

            match &current.node_type {
                NodeType::Leaf { primitive_index } => {
                    let intersection: Option<HitInfo> =
                        self.objects[*primitive_index as usize].intersect(r, t_max);
                    if intersection.is_some() {
                        t_max = intersection.as_ref().unwrap().t;
                        closest = intersection.zip(Some(&self.objects[*primitive_index as usize]));
                    }
                }
                NodeType::Branch {
                    split_axis,
                    left,
                    right,
                } => {
                    let (a, b): (&Box<BVHNode>, &Box<BVHNode>) =
                        if r.direction[*split_axis as usize] > 0.0 {
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
                _ => unsafe {
                    std::hint::unreachable_unchecked();
                },
            }
        }

        closest
    }
    pub fn any_intersect(&self, r: &Ray, t_max: f64) -> bool
    {
        let mut stack: Vec<&BVHNode> = vec![&self.bvh];

        while !stack.is_empty() {
            let current: &BVHNode = stack.pop().unwrap();

            if !current.bounding_box.intersect(r, t_max) {
                continue;
            }

            match &current.node_type {
                NodeType::Leaf { primitive_index } => {
                    if self.objects[*primitive_index as usize].intersect_bool(r, t_max) {
                        return true;
                    }
                }
                NodeType::Branch {
                    split_axis: _,
                    left,
                    right,
                } => {
                    stack.push(left);
                    stack.push(right);
                }
                _ => unsafe {
                    std::hint::unreachable_unchecked();
                },
            }
        }

        false
    }
}

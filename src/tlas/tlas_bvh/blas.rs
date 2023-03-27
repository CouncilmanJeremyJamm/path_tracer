use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use bumpalo::Bump;
use id_arena::{Arena, Id};
use rayon::prelude::ParallelIterator;

use primitive::model::{Model, Vertex, VertexRef};
use primitive::Triangle;

use crate::tlas::tlas_bvh::blas::blas_bvh::boundingbox::HasBox;
use crate::tlas::tlas_bvh::blas::blas_bvh::{BLASNode, BLASNodeType, PrimitiveInfo};
use crate::{HitInfo, Material, MaterialTrait, Ray};

#[macro_use]
pub mod blas_bvh;
pub mod primitive;

#[derive(Clone)]
pub enum TokenType
{
    Comment,
    Vertex,
    Normal,
    Texture,
    Face,
    Group,
    Material,
}

static TOKEN_TYPE: phf::Map<&'static str, TokenType> = phf::phf_map! {
    "#" => TokenType::Comment,
    "v" => TokenType::Vertex,
    "vn" => TokenType::Normal,
    "vt" => TokenType::Texture,
    "f" => TokenType::Face,
    "g" => TokenType::Group,
    "usemtl" => TokenType::Material,
};

pub fn parse_token(keyword: &str) -> Option<TokenType> { TOKEN_TYPE.get(keyword).cloned() }

pub fn load_obj(path: &std::path::Path) -> Vec<Vertex>
{
    let mut normals: Vec<glam::Vec3A> = vec![glam::Vec3A::ZERO];
    let mut positions: Vec<glam::Vec3A> = vec![glam::Vec3A::ZERO];

    let mut vertices: Vec<Vertex> = Vec::new();

    let file: File = File::open(path).unwrap();
    let lines = BufReader::new(file).lines().filter_map(|l| l.ok());
    let timer: Instant = Instant::now();
    for line in lines
    {
        let tokens: Vec<&str> = line.split_whitespace().collect();

        match parse_token(tokens[0])
        {
            Some(TokenType::Vertex) =>
            {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                positions.push(glam::Vec3A::new(x, y, z));
            }
            Some(TokenType::Normal) =>
            {
                let x: f32 = tokens[1].parse().unwrap();
                let y: f32 = tokens[2].parse().unwrap();
                let z: f32 = tokens[3].parse().unwrap();

                normals.push(glam::Vec3A::new(x, y, z).normalize());
            }
            Some(TokenType::Face) =>
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
            //Comments, textures, unknown
            {
                continue;
            }
        }
    }

    println!("loading model took {} ms", timer.elapsed().as_millis());
    vertices
}

pub(crate) fn push_to_stack<N>(r: &Ray, t_max: f32, stack: &mut Vec<(Id<N>, f32), &Bump>, arena: &Arena<N>, left: Id<N>, right: Id<N>)
where
    N: HasBox,
{
    let intersect_left = arena[left].get_box().intersect_t(r, t_max);
    let intersect_right = arena[right].get_box().intersect_t(r, t_max);

    if let (Some(t_enter_left), Some(t_enter_right)) = (intersect_left, intersect_right)
    {
        stack.reserve(2);
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

pub(crate) struct BLAS
{
    pub primitives: Arena<Triangle>,
    pub material: Material,
    pub root_id: Id<BLASNode>,
    pub blas_node_arena: Arena<BLASNode>,
}

impl<'a> BLAS
{
    pub fn new(model: Model<'a>) -> Self
    {
        let timer: Instant = Instant::now();

        let vertices: Vec<Vertex> = load_obj(model.file_path);
        let mut primitives: Arena<Triangle> = Arena::with_capacity(vertices.len() / 3);

        vertices.array_chunks::<3>().for_each(|v| {
            primitives.alloc(Triangle::new(v));
        });

        let mut primitive_info: Vec<PrimitiveInfo> = primitives
            .par_iter()
            .map(|(id, p): (Id<Triangle>, &Triangle)| -> PrimitiveInfo { PrimitiveInfo::new(p.create_bounding_box(), id) })
            .collect();

        let mut blas_arena: Arena<BLASNode> = Arena::new();
        let root_id: Id<BLASNode> = BLASNode::generate_blas(&mut blas_arena, primitive_info.as_mut_slice(), 4);

        println!("BLAS - {:?}: \t{:?}", model.file_path.file_name().unwrap(), timer.elapsed());

        Self {
            primitives,
            material: model.material,
            root_id,
            blas_node_arena: blas_arena,
        }
    }

    pub fn generate_lights(&self, blas_id: Id<BLAS>) -> Vec<(Id<BLAS>, Id<Triangle>, f32)>
    {
        self.primitives
            .iter()
            .map(|(id, p): (Id<Triangle>, &Triangle)| {
                let weight: f32 = p.area() * self.material.get_emitted().length();
                (blas_id, id, weight)
            })
            .collect()
    }

    pub fn intersect(&self, bump: &Bump, r: &Ray, mut t_max: f32) -> Option<(HitInfo, Id<Triangle>)>
    {
        let mut stack: Vec<(Id<BLASNode>, f32), _> = Vec::new_in(bump);
        stack.push((self.root_id, 0.0));
        let mut closest: Option<(HitInfo, Id<Triangle>)> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &self.blas_node_arena[current].node_type
            {
                BLASNodeType::Branch { left, right } => push_to_stack(r, t_max, &mut stack, &self.blas_node_arena, *left, *right),
                BLASNodeType::LeafSingle { primitive_id } =>
                //Fast path for single child
                {
                    let primitive: &Triangle = &self.primitives[*primitive_id];
                    if let Some(intersection) = primitive.intersect(r, t_max, t_enter)
                    {
                        t_max = intersection.t;
                        closest = Some((intersection, *primitive_id));
                        //println!("{} {}", t_enter, t_max);
                    }
                }
                BLASNodeType::Leaf { primitive_ids } =>
                {
                    for id in primitive_ids.as_ref()
                    {
                        if let Some(intersection) = self.primitives[*id].intersect(r, t_max, t_enter)
                        {
                            t_max = intersection.t;
                            closest = Some((intersection, *id));
                        }
                    }
                }
            }
        }

        closest
    }
    pub fn any_intersect(&self, bump: &Bump, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<Id<BLASNode>, _> = Vec::new_in(bump);
        stack.push(self.root_id);

        while let Some(current) = stack.pop()
        {
            if let Some(t_enter) = self.blas_node_arena[current].bounding_box.intersect_t(r, t_max)
            {
                match &self.blas_node_arena[current].node_type
                {
                    BLASNodeType::Branch { left, right } =>
                    {
                        stack.reserve(2);
                        stack.push(*left);
                        stack.push(*right);
                    }
                    BLASNodeType::LeafSingle { primitive_id } =>
                    //Fast path for single child
                    {
                        if self.primitives[*primitive_id].intersect_bool(r, t_max, t_enter)
                        {
                            return true;
                        }
                    }
                    BLASNodeType::Leaf { primitive_ids } =>
                    {
                        if primitive_ids.iter().any(|id| self.primitives[*id].intersect_bool(r, t_max, t_enter))
                        {
                            return true;
                        }
                    }
                }
            }
        }

        false
    }
}

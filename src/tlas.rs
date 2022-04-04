use std::cmp::Ordering;

use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::tlas::blas::bvh::boundingbox::{surrounding_box, AABB};
use crate::tlas::blas::BLAS;
use crate::{HitInfo, Material, Model, Ray, Triangle};

pub mod blas;

struct BLASInfo
{
    bounding_box: AABB,
    blas_index: u8,
}

impl BLASInfo
{
    pub fn new(bounding_box: AABB, blas_index: u8) -> Self { Self { bounding_box, blas_index } }
}

fn create_bounding_box(object_info: &[BLASInfo]) -> AABB
{
    object_info
        .iter()
        .fold(AABB::identity(), |a: AABB, b: &BLASInfo| surrounding_box(&a, &b.bounding_box))
}

enum TLASNodeType
{
    Leaf
    {
        blas_index: u8
    },
    Branch
    {
        left: Box<TLASNode>, right: Box<TLASNode>
    },
}

struct TLASNode
{
    bounding_box: AABB,
    node_type: TLASNodeType,
}

impl TLASNode
{
    fn generate_tlas(blas_info: &mut [BLASInfo]) -> Self
    {
        let blas_span: usize = blas_info.len();
        debug_assert!(blas_span > 0);

        if blas_span == 1
        {
            Self {
                bounding_box: blas_info[0].bounding_box,
                node_type: TLASNodeType::Leaf {
                    blas_index: blas_info[0].blas_index,
                },
            }
        }
        else
        {
            let bounding_box: AABB = create_bounding_box(blas_info);

            //Find longest axis
            let box_length: glam::Vec3A = bounding_box.length();
            let max_length: f32 = box_length.max_element();

            let split_axis: u8 = if box_length.x == max_length
            {
                0u8
            }
            else if box_length.y == max_length
            {
                1u8
            }
            else
            {
                2u8
            };

            let comparator = |a: &BLASInfo, b: &BLASInfo| -> Ordering { a.bounding_box.compare(&b.bounding_box, split_axis) };
            blas_info.sort_unstable_by(comparator);

            let (left_info, right_info) = blas_info.split_at_mut(blas_span / 2);
            let (left, right) = rayon::join(|| Box::new(Self::generate_tlas(left_info)), || Box::new(Self::generate_tlas(right_info)));

            Self {
                bounding_box,
                node_type: TLASNodeType::Branch { left, right },
            }
        }
    }
}

pub struct TLAS<'a>
{
    blas_vec: Vec<BLAS<'a>>,
    num_primitives: usize,
    bvh: TLASNode,
}

impl<'a> TLAS<'a>
{
    pub fn new(models: Vec<Model<'a>>) -> Self
    {
        let blas_vec: Vec<BLAS<'a>> = models.par_iter().map(BLAS::new).collect();

        let num_primitives: usize = blas_vec.iter().map(|blas| blas.objects.len()).sum();

        let mut blas_info: Vec<BLASInfo> = blas_vec
            .par_iter()
            .enumerate()
            .map(|tuple: (usize, &BLAS)| -> BLASInfo { BLASInfo::new(tuple.1.bvh.bounding_box, tuple.0 as u8) })
            .collect();

        let bvh: TLASNode = TLASNode::generate_tlas(blas_info.as_mut_slice());

        Self {
            blas_vec,
            num_primitives,
            bvh,
        }
    }

    pub fn random_primitive(&self, rng: &mut TlsWyRand) -> (usize, &(dyn Material), &Triangle)
    {
        let blas: &BLAS = &self.blas_vec[rng.generate_range(0..self.blas_vec.len())];

        (
            self.num_primitives,
            blas.material,
            &blas.objects[rng.generate_range(0..blas.objects.len())],
        )
    }

    pub fn intersect(&self, r: &Ray, mut t_max: f32) -> Option<(HitInfo, &Triangle, &(dyn Material))>
    {
        if !self.bvh.bounding_box.intersect(r, t_max)
        {
            return None;
        }

        let mut stack: Vec<(&TLASNode, f32)> = vec![(&self.bvh, 0.0)];
        let mut closest: Option<(HitInfo, &Triangle, &(dyn Material))> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &current.node_type
            {
                TLASNodeType::Branch { left, right } =>
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
                TLASNodeType::Leaf { blas_index } =>
                {
                    let blas: &BLAS = &self.blas_vec[*blas_index as usize];
                    if let Some((hit_info, triangle, material)) = blas.intersect(r, t_max)
                    {
                        t_max = hit_info.t;
                        closest = Some((hit_info, triangle, material));
                    }
                }
            }
        }

        closest
    }
    pub fn any_intersect(&self, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<&TLASNode> = vec![&self.bvh];

        while let Some(current) = stack.pop()
        {
            if !current.bounding_box.intersect(r, t_max)
            {
                continue;
            }

            match &current.node_type
            {
                TLASNodeType::Branch { left, right } =>
                {
                    stack.push(left);
                    stack.push(right);
                }
                TLASNodeType::Leaf { blas_index } =>
                {
                    if self.blas_vec[*blas_index as usize].any_intersect(r, t_max)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

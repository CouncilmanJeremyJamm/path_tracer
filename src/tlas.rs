use std::time::Instant;

use bumpalo::Bump;
use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use tlas_bvh::{BLASInfo, TLASNode, TLASNodeType};

use crate::tlas::blas::{push_to_stack, BLAS};
use crate::{HitInfo, Material, Model, Ray, Triangle};

pub mod blas;
mod tlas_bvh;

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
        let timer: Instant = Instant::now();

        let blas_vec: Vec<BLAS<'a>> = models.par_iter().map(BLAS::new).collect();

        let num_primitives: usize = blas_vec.iter().map(|blas| blas.primitives.len()).sum();

        let mut blas_info: Vec<BLASInfo> = blas_vec
            .par_iter()
            .enumerate()
            .map(|tuple: (usize, &BLAS)| -> BLASInfo { BLASInfo::new(tuple.1.bvh.bounding_box, tuple.0 as u8) })
            .collect();

        let bvh: TLASNode = TLASNode::generate_tlas(blas_info.as_mut_slice());

        println!("TLAS: {:?}\n", timer.elapsed());

        Self {
            blas_vec,
            num_primitives,
            bvh,
        }
    }

    pub fn random_primitive(&self, rng: &mut TlsWyRand) -> (usize, &Material, &Triangle)
    {
        let blas: &BLAS = &self.blas_vec[rng.generate_range(0..self.blas_vec.len())];

        (
            self.num_primitives,
            blas.material,
            &blas.primitives[rng.generate_range(0..blas.primitives.len())],
        )
    }

    pub fn intersect(&self, bump: &Bump, r: &Ray, mut t_max: f32) -> Option<(HitInfo, &Triangle, &Material)>
    {
        if !self.bvh.bounding_box.intersect(r, t_max)
        {
            return None;
        }

        let mut stack: Vec<(&TLASNode, f32), _> = Vec::with_capacity_in(1, bump);
        stack.push((&self.bvh, 0.0));

        let mut closest: Option<(HitInfo, &Triangle, &Material)> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &current.node_type
            {
                TLASNodeType::Branch { left, right } => push_to_stack(r, t_max, &mut stack, left, right),
                TLASNodeType::Leaf { blas_index } =>
                {
                    let blas: &BLAS = &self.blas_vec[*blas_index as usize];
                    if let Some((hit_info, triangle, material)) = blas.intersect(bump, r, t_max)
                    {
                        t_max = hit_info.t;
                        closest = Some((hit_info, triangle, material));
                    }
                }
            }
        }

        closest
    }
    pub fn any_intersect(&self, bump: &Bump, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<&TLASNode, _> = Vec::with_capacity_in(1, bump);
        stack.push(&self.bvh);

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
                    if self.blas_vec[*blas_index as usize].any_intersect(bump, r, t_max)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

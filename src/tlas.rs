use nanorand::tls::TlsWyRand;
use nanorand::Rng;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use tlas_bvh::{BLASInfo, TLASNode, TLASNodeType};

use crate::tlas::blas::BLAS;
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

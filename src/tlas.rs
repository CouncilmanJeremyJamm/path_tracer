use std::time::Instant;

use bumpalo::Bump;
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use tlas_bvh::{BLASInfo, TLASNode, TLASNodeType};

use crate::scene::light_sampler::LightSampler;
use crate::tlas::blas::{push_to_stack, BLAS};
use crate::{HitInfo, Material, Model, Ray, Triangle};

pub mod blas;
mod tlas_bvh;

pub(crate) struct TLAS<'a>
{
    pub blas_vec: Vec<BLAS<'a>>,
    bvh: TLASNode,
}

impl<'a> TLAS<'a>
{
    pub fn new(models: Vec<Model<'a>>) -> Self
    {
        let timer: Instant = Instant::now();

        let blas_vec: Vec<BLAS<'a>> = models.par_iter().map(BLAS::new).collect();

        let mut blas_info: Vec<BLASInfo> = blas_vec
            .par_iter()
            .enumerate()
            .zip_eq(models.into_par_iter())
            .map(|(tuple, model): ((usize, &BLAS), Model)| -> BLASInfo { BLASInfo::new(tuple.1.bvh.bounding_box, tuple.0 as u8, model.matrices) })
            .collect();

        let bvh: TLASNode = TLASNode::generate_tlas(blas_info.as_mut_slice());

        println!("TLAS: {:?}\n", timer.elapsed());

        Self { blas_vec, bvh }
    }

    pub fn generate_lights(&self) -> LightSampler
    {
        let lights: Vec<_> = self
            .blas_vec
            .iter()
            .enumerate()
            .flat_map(|(i, blas)| blas.generate_lights(i as u8))
            .collect();

        LightSampler::new(lights)
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
                TLASNodeType::Leaf {
                    blas_index,
                    matrix,
                    inv_matrix,
                } =>
                {
                    let ray: Ray = r.transform(inv_matrix);

                    let blas: &BLAS = &self.blas_vec[*blas_index as usize];
                    if let Some((hit_info, triangle, material)) = blas.intersect(bump, &ray, t_max)
                    {
                        t_max = hit_info.t;
                        closest = Some((
                            HitInfo {
                                normal: matrix.transform_vector3a(hit_info.normal),
                                ..hit_info
                            },
                            triangle,
                            material,
                        ));
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
                TLASNodeType::Leaf { blas_index, inv_matrix, .. } =>
                {
                    let ray: Ray = r.transform(inv_matrix);

                    if self.blas_vec[*blas_index as usize].any_intersect(bump, &ray, t_max)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

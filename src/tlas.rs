use std::time::Instant;

use bumpalo::Bump;
use id_arena::{Arena, Id};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use tlas_bvh::blas::{push_to_stack, BLAS};
use tlas_bvh::{BLASInfo, TLASNode, TLASNodeType};

use crate::scene::light_sampler::LightSampler;
use crate::{HitInfo, Material, Model, Ray, Triangle};

pub(crate) mod tlas_bvh;

pub(crate) struct TLAS
{
    pub blas_arena: Arena<BLAS>,
    tlas_arena: Arena<TLASNode>,
    tlas_root: Id<TLASNode>,
}

impl<'a> TLAS
{
    pub fn new(models: Vec<Model<'a>>) -> Self
    {
        let timer: Instant = Instant::now();

        let matrices_vec: Vec<Vec<glam::Affine3A>> = models.iter().map(|m| m.matrices.clone()).collect();

        let mut blas_arena: Arena<BLAS> = Arena::with_capacity(models.len());
        models.into_iter().for_each(|m: Model| {
            blas_arena.alloc(BLAS::new(m));
        });

        let blas_info: Vec<BLASInfo> = blas_arena
            .par_iter()
            .zip_eq(matrices_vec.into_par_iter())
            .map(|((blas_id, blas), matrices): ((Id<BLAS>, &BLAS), Vec<glam::Affine3A>)| -> BLASInfo {
                BLASInfo::new(blas.blas_node_arena[blas.root_id].bounding_box, blas_id, matrices)
            })
            .collect();

        let mut tlas_arena: Arena<TLASNode> = Arena::new();
        let tlas_root: Id<TLASNode> = TLASNode::generate_tlas(&mut tlas_arena, blas_info.as_slice());

        println!("TLAS: {:?}\n", timer.elapsed());

        Self {
            blas_arena,
            tlas_arena,
            tlas_root,
        }
    }

    pub fn generate_lights(&self) -> LightSampler
    {
        let lights: Vec<(Id<BLAS>, Id<Triangle>, f32)> = self
            .blas_arena
            .iter()
            .flat_map(|(id, blas): (Id<BLAS>, &BLAS)| blas.generate_lights(id))
            .collect();

        LightSampler::new(lights)
    }

    pub fn intersect(&self, bump: &Bump, r: &Ray, mut t_max: f32) -> Option<(HitInfo, Id<BLAS>, &Triangle, &Material)>
    {
        if !self.tlas_arena[self.tlas_root].bounding_box.intersect(r, t_max)
        {
            return None;
        }

        let mut stack: Vec<(Id<TLASNode>, f32), _> = Vec::with_capacity_in(1, bump);
        stack.push((self.tlas_root, 0.0));

        let mut closest: Option<(HitInfo, Id<BLAS>, &glam::Affine3A, Id<Triangle>)> = None;

        while let Some((current, t_enter)) = stack.pop()
        {
            if t_enter > t_max
            {
                continue;
            }

            match &self.tlas_arena[current].node_type
            {
                TLASNodeType::Branch { left, right } => push_to_stack(r, t_max, &mut stack, &self.tlas_arena, *left, *right),
                TLASNodeType::Leaf { blas_id, matrix, inv_matrix } =>
                {
                    let ray: Ray = r.transform(inv_matrix);

                    let blas: &BLAS = &self.blas_arena[*blas_id];
                    if let Some((hit_info, primitive_id)) = blas.intersect(bump, &ray, t_max)
                    {
                        t_max = hit_info.t;
                        // Defer transform of normal until the end
                        closest = Some((hit_info, *blas_id, matrix, primitive_id));
                    }
                }
            }
        }

        closest.map(|(mut hit_info, blas_id, matrix, primitive_id)| {
            // Transform the recorded normal using the transformation matrix from the parent instance
            hit_info.normal = matrix.transform_vector3a(hit_info.normal);

            let blas: &BLAS = &self.blas_arena[blas_id];
            (hit_info, blas_id, &blas.primitives[primitive_id], &blas.material)
        })
    }
    pub fn any_intersect(&self, bump: &Bump, r: &Ray, t_max: f32) -> bool
    {
        let mut stack: Vec<Id<TLASNode>, _> = Vec::with_capacity_in(1, bump);
        stack.push(self.tlas_root);

        while let Some(current) = stack.pop()
        {
            if !self.tlas_arena[current].bounding_box.intersect(r, t_max)
            {
                continue;
            }

            match &self.tlas_arena[current].node_type
            {
                TLASNodeType::Branch { left, right } =>
                {
                    stack.reserve(2);
                    stack.push(*left);
                    stack.push(*right);
                }
                TLASNodeType::Leaf { blas_id, inv_matrix, .. } =>
                {
                    let ray: Ray = r.transform(inv_matrix);

                    if self.blas_arena[*blas_id].any_intersect(bump, &ray, t_max)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }
}

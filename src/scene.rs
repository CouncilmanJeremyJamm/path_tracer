use nanorand::tls::TlsWyRand;

use light_sampler::LightSampler;

use crate::scene::light_sampler::LightItem;
use crate::tlas::tlas_bvh::blas::BLAS;
use crate::MaterialTrait;
use crate::{Material, Model, Triangle, TLAS};

pub mod light_sampler;

pub(crate) struct Scene
{
    pub world: TLAS,
    pub lights: TLAS,
    pub light_sampler: LightSampler,
}

impl<'a> Scene
{
    pub fn new(models: Vec<Model<'a>>) -> Self
    {
        let lights_models: Vec<Model> = models.iter().cloned().filter(|m| m.material.is_emissive()).collect();

        let world: TLAS = TLAS::new(models);
        let lights: TLAS = TLAS::new(lights_models);

        let light_sampler: LightSampler = lights.generate_lights();

        Self {
            world,
            lights,
            light_sampler,
        }
    }

    pub fn sample_lights(&self, rng: &mut TlsWyRand) -> (&Material, &Triangle, f32)
    {
        let light_item: LightItem = self.light_sampler.sample(rng);

        let blas: &BLAS = &self.lights.blas_arena[light_item.blas_id];
        let primitive: &Triangle = &blas.primitives[light_item.primitive_id];

        (&blas.material, primitive, light_item.pdf)
    }
}

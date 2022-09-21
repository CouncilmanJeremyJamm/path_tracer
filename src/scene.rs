use nanorand::tls::TlsWyRand;

use light_sampler::LightSampler;

use crate::scene::light_sampler::LightItem;
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

        let light_sampler = lights.generate_lights();

        Self {
            world,
            lights,
            light_sampler,
        }
    }

    pub fn sample_lights(&self, rng: &mut TlsWyRand) -> (&Material, &Triangle, f32)
    {
        let light_item: LightItem = self.light_sampler.sample(rng);

        let blas = &self.lights.blas_vec[light_item.blas_index as usize];
        let primitive = &blas.primitives[light_item.primitive_index as usize];

        (&blas.material, primitive, light_item.pdf)
    }
}

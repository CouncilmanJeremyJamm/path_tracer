use id_arena::Id;
use nanorand::tls::TlsWyRand;
use nanorand::Rng;

use crate::tlas::tlas_bvh::blas::BLAS;
use crate::{Material, MaterialTrait, Triangle};

#[derive(Copy, Clone, Debug)]
pub(crate) struct LightItem
{
    pub blas_id: Id<BLAS>,
    pub primitive_id: Id<Triangle>,
    pub(crate) pdf: f32,
}

impl LightItem
{
    pub fn new(blas_id: Id<BLAS>, primitive_id: Id<Triangle>, pdf: f32) -> Self { Self { blas_id, primitive_id, pdf } }
}

#[derive(Debug)]
pub(crate) struct LightSampler
{
    lights: Vec<LightItem>,
    lights_cdf: Vec<f32>,
    max: f32,
}

impl LightSampler
{
    pub fn sample(&self, rng: &mut TlsWyRand) -> LightItem
    {
        let x: f32 = rng.generate();
        let index: usize = self.lights_cdf.binary_search_by(|a| a.total_cmp(&x)).unwrap_or_else(|i| i);

        self.lights[index]
    }

    pub fn get_sample_pdf(&self, primitive: &Triangle, material: &Material) -> f32 { primitive.area() * material.get_emitted().length() / self.max }

    pub fn new(lights_data: Vec<(Id<BLAS>, Id<Triangle>, f32)>) -> Self
    {
        let max: f32 = lights_data.iter().map(|&(_, _, weight)| weight).sum();

        let lights: Vec<LightItem> = lights_data
            .iter()
            .map(|&(blas_index, primitive_id, weight)| LightItem::new(blas_index, primitive_id, weight / max))
            .collect();

        let lights_cdf: Vec<f32> = lights
            .iter()
            .scan(0.0, |state: &mut f32, &li: &LightItem| {
                *state += li.pdf;
                Some(*state)
            })
            .collect();

        //println!("{:?}", lights_cdf);

        Self { lights, lights_cdf, max }
    }
}

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::prelude::IntoParallelRefMutIterator;

const DIRECTIONS: [u32; 32] = [
    0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000, 0xaa000000, 0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
    0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000, 0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
    0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc, 0xaaaaaaaa, 0xffffffff,
];

/// A wrapper struct around a vector of `u32` Sobol points,
/// containing helper functions to return shuffled and scrambled `f32` points
/// # Parameters
/// * `n` - Number of points in the sequence.
/// This can be higher than the number of samples per pixel,
/// as each index is shuffled before reading
pub struct SobolSampler<const N: usize>
{
    sobol: [glam::UVec2; N],
}

impl<const N: usize> SobolSampler<N>
{
    /// Computes a single Sobol point
    fn sobol(index: u32) -> u32
    {
        DIRECTIONS.iter().enumerate().fold(0u32, |x, (bit, &direction)| {
            let mask: u32 = (index >> bit) & 1u32;
            x ^ (mask * direction)
        })
    }

    /// Initialises the struct with a vector of Sobol points
    pub fn generate_sobol() -> Self
    {
        assert!(u32::try_from(N).is_ok(), "Too many points");
        let mut sobol: [glam::UVec2; N] = [glam::UVec2::ZERO; N];

        sobol.par_iter_mut().enumerate().for_each(|(i, point)| {
            let index: u32 = i as u32;

            let x: u32 = index.reverse_bits();
            let y: u32 = Self::sobol(index);

            *point = glam::UVec2::new(x, y);
        });

        Self { sobol }
    }

    /// An improved Laine-Karras hash.
    ///
    /// Credit to: https://psychopath.io/post/2021_01_30_building_a_better_lk_hash
    fn lk_hash(mut x: u32, seed: u32) -> u32
    {
        // x ^= x * 0x3d20adea;
        // x += seed;
        // x *= (seed >> 16) | 1;
        // x ^= x * 0x05526c56;
        // x ^= x * 0x53a22864;

        x ^= x.wrapping_mul(0x3d20adea);
        x = x.wrapping_add(seed);
        x = x.wrapping_mul(seed.wrapping_shr(16) | 1);
        x ^= x.wrapping_mul(0x05526c56);
        x ^= x.wrapping_mul(0x53a22864);

        x
    }

    /// Scrambles a `u32` in base-2 using an LK hash
    fn scramble_base2(x: u32, seed: u32) -> u32 { Self::lk_hash(x.reverse_bits(), seed).reverse_bits() }

    /// A fast 32 bit 2-round hash, used to generate multiple seeds from sequential seeds
    ///
    /// Credit to: https://github.com/skeeto/hash-prospector
    fn low_bias_hash(mut x: u32) -> u32
    {
        // x ^= x >> 16;
        // x *= 0x21f0aaad;
        // x ^= x >> 15;
        // x *= 0xd35a2d97;
        // x ^= x >> 15;

        x ^= x.wrapping_shr(16);
        x = x.wrapping_mul(0x21f0aaad);
        x ^= x.wrapping_shr(15);
        x = x.wrapping_mul(0xd35a2d97);
        x ^= x.wrapping_shr(15);

        x
    }

    /// Returns a shuffled-scrambled Sobol point inside the unit square.
    ///
    /// Based almost entirely on an implementation by **Andrew Helmer**:
    /// https://www.reddit.com/r/GraphicsProgramming/comments/l1go2r/owenscrambled_sobol_02_sequences_shadertoy/
    pub fn get_ss_sobol(&self, index: u32, seed: u32) -> glam::Vec2
    {
        let x_seed: u32 = Self::low_bias_hash(seed);
        let y_seed: u32 = Self::low_bias_hash(seed + 1);
        let shuffle_seed: u32 = Self::low_bias_hash(seed + 2);

        let shuffled_index: u32 = Self::scramble_base2(index, shuffle_seed);
        let sobol_pt: glam::UVec2 = self.sobol[shuffled_index as usize % N];

        let x: u32 = Self::scramble_base2(sobol_pt.x, x_seed);
        let y: u32 = Self::scramble_base2(sobol_pt.y, y_seed);

        let p: glam::Vec2 = glam::Vec2::new(x as f32, y as f32) / (u32::MAX as f32);
        debug_assert!((0.0..=1.0).contains(&p.x));
        debug_assert!((0.0..=1.0).contains(&p.y));

        p
    }
}

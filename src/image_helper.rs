use std::path::Path;

use glam::const_mat3a;
use image::ImageError;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::image_helper::tonemapping::gt_tonemap_vector;

mod tonemapping;

/// Helper struct for loading/writing images.
///
/// External images use gamma 2.2, internal data uses linear RGB
pub struct ImageHelper
{
    data: Vec<glam::Vec3A>,
    pub dimensions: glam::UVec2,
}

impl ImageHelper
{
    pub fn new(data: Vec<glam::Vec3A>, dimensions: glam::UVec2) -> Self { Self { data, dimensions } }

    /// Loads the image from the specified path,
    /// converting from gamma 2.2 to linear RGB
    pub fn load_image<P: AsRef<Path>>(path: P) -> Result<ImageHelper, ImageError>
    {
        let img = image::io::Reader::open(path)?.decode()?.into_rgb32f();
        let data = img
            .array_chunks::<3>()
            .map(|rgb| glam::Vec3A::from_slice(rgb).powf(2.2))
            .collect::<Vec<_>>();
        Ok(ImageHelper::new(data, glam::UVec2::from(img.dimensions())))
    }

    /// Writes the internal image data to the specified path,
    /// converting from linear RGB to gamma 2.2
    pub fn write_image<P: AsRef<Path>>(&self, path: P) -> Result<(), ImageError>
    {
        println!("Converting data...");
        let data: Vec<u8> = self
            .data
            .par_iter()
            .flat_map_iter(|&a| {
                let out: glam::Vec3A = gt_tonemap_vector(a, 1.0, 1.0, 0.22, 0.4, 1.33, 0.0).powf(1.0 / 2.2) * 255.0;

                [out.x as u8, out.y as u8, out.z as u8]
            })
            .collect();

        println!("Saving output...");
        image::save_buffer(
            path.as_ref(),
            data.as_slice(),
            self.dimensions.x,
            self.dimensions.y,
            image::ColorType::Rgb8,
        )
    }

    /// Returns the pixel at absolute coordinates (x, y), with coordinate wrapping in both axes
    fn get_pixel(&self, x: u32, y: u32) -> glam::Vec3A
    {
        let _x = x % self.dimensions.x;
        let _y = y % self.dimensions.y;

        let i: usize = (_y * self.dimensions.x + _x) as usize;
        self.data[i]
    }

    /// Samples the image data at UV coordinates (u,v) using bilinear interpolation
    pub fn get_pixel_bilinear(&self, u: f32, v: f32) -> glam::Vec3A
    {
        let x: f32 = (self.dimensions.x as f32) * u;
        let y: f32 = (self.dimensions.y as f32) * v;

        let x0: u32 = x as u32;
        let y0: u32 = y as u32;

        let x_fract: f32 = x.fract();
        let y_fract: f32 = y.fract();

        let c_00: glam::Vec3A = self.get_pixel(x0, y0);
        let c_01: glam::Vec3A = self.get_pixel(x0, y0 + 1);
        let c_10: glam::Vec3A = self.get_pixel(x0 + 1, y0);
        let c_11: glam::Vec3A = self.get_pixel(x0 + 1, y0 + 1);

        (1.0 - x_fract) * (1.0 - y_fract) * c_00 + (1.0 - x_fract) * y_fract * c_01 + x_fract * (1.0 - y_fract) * c_10 + x_fract * y_fract * c_11
    }
}

use std::path::Path;

use image::ImageError;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

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
            .flat_map_iter(|a| {
                let fvec = a.clamp(glam::Vec3A::ZERO, glam::Vec3A::ONE).powf(1.0 / 2.2) * 255.0;
                [fvec.x as u8, fvec.y as u8, fvec.z as u8]
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

    /// Returns the pixel at (x, y), with coordinate wrapping in both axes
    pub fn get_pixel(&self, x: u32, y: u32) -> glam::Vec3A
    {
        let _x = x % self.dimensions.x;
        let _y = y % self.dimensions.y;

        let i: usize = (_y * self.dimensions.x + _x) as usize;
        self.data[i]
    }
}

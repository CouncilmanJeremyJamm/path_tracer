pub struct Ray
{
    pub origin: glam::Vec3A,
    pub direction: glam::Vec3A,
    pub inv_direction: glam::Vec3A,
}

impl Ray
{
    pub fn new(o: glam::Vec3A, d: glam::Vec3A) -> Self
    {
        debug_assert!(d.is_normalized());
        Self {
            origin: o,
            direction: d,
            inv_direction: d.recip(),
        }
    }

    pub fn at(&self, t: f32) -> glam::Vec3A { self.direction.mul_add(glam::Vec3A::splat(t), self.origin) }
}

pub struct Ray
{
    pub origin: glam::DVec3,
    pub direction: glam::DVec3,
    pub inv_direction: glam::DVec3,
}

impl Ray
{
    pub fn new(o: glam::DVec3, d: glam::DVec3) -> Self
    {
        Self {
            origin: o,
            direction: d,
            inv_direction: d.recip(),
        }
    }

    pub fn at(&self, t: f64) -> glam::DVec3
    {
        self.origin + (self.direction * t)
    }
}

pub fn generate_onb(normal: glam::Vec3A) -> glam::Mat3A
{
    let c2: glam::Vec3A = glam::Vec3A::normalize(normal);
    let (c0, c1): (glam::Vec3A, glam::Vec3A) = glam::Vec3A::any_orthonormal_pair(&c2);
    glam::Mat3A::from_cols(c0, c1, c2)
}

pub fn generate_onb_ggx(v: glam::Vec3A) -> glam::Mat3A
{
    let t1: glam::Vec3A = if v.z < 0.9999
    {
        v.cross(glam::Vec3A::Z).normalize()
    }
    else
    {
        glam::Vec3A::X
    };
    let t2: glam::Vec3A = t1.cross(v);

    glam::Mat3A::from_cols(t1, t2, v)
}

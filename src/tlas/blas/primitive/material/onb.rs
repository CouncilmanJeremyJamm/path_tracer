pub fn generate_onb(normal: glam::Vec3A) -> glam::Mat3A
{
    debug_assert!(normal.is_normalized());

    let (c0, c1): (glam::Vec3A, glam::Vec3A) = glam::Vec3A::any_orthonormal_pair(&normal);
    glam::Mat3A::from_cols(c0, c1, normal)
}

pub fn generate_onb_ggx(v: glam::Vec3A) -> glam::Mat3A
{
    debug_assert!(v.is_normalized());

    if v.z > 0.99999
    {
        glam::Mat3A::from_cols(glam::Vec3A::X, -glam::Vec3A::Y, glam::Vec3A::Z)
    }
    else
    {
        let t1: glam::Vec3A = v.cross(glam::Vec3A::Z).normalize();
        let t2: glam::Vec3A = t1.cross(v);

        debug_assert!(t1.is_normalized());
        debug_assert!(t2.is_normalized());

        glam::Mat3A::from_cols(t1, t2, v)
    }
}

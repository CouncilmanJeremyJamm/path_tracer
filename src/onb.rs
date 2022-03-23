pub fn generate_onb(normal: glam::DVec3) -> glam::DMat3
{
    let c2: glam::DVec3 = glam::DVec3::normalize(normal);
    let (c0, c1): (glam::DVec3, glam::DVec3) = glam::DVec3::any_orthonormal_pair(&c2);
    glam::DMat3::from_cols(c0, c1, c2)
}

pub fn generate_onb_ggx(v: glam::DVec3) -> glam::DMat3
{
    let t1: glam::DVec3 = if v.z < 0.9999 {
        v.cross(glam::DVec3::new(0.0, 0.0, 1.0)).normalize()
    } else {
        glam::DVec3::new(1.0, 0.0, 0.0)
    };
    let t2: glam::DVec3 = t1.cross(v);

    glam::DMat3::from_cols(t1, t2, v)
}

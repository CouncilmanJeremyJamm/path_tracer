/// Defines the linear section of the tonemap curve
fn gt_linear(x: f32, a: f32, m: f32) -> f32 { m + a * (x - m) }

/// Defines the 'toe' of the tonemap curve
fn gt_toe(x: f32, m: f32, c: f32, b: f32) -> f32 { m * (x / m).powf(c) + b }

/// Defines the 'shoulder' of the tonemap curve
fn gt_shoulder(x: f32, p: f32, a: f32, m: f32, l0: f32) -> f32
{
    let s0: f32 = m + l0;
    let s1: f32 = m + a * l0;

    let c2: f32 = a * p / (p - s1);

    p - (p - s1) * (-c2 * (x - s0) / p).exp()
}

/// Linear interpolation between 0 and 1 across the range [e0, e1]
/// Denoted as `h` in the original source
fn gt_lerp(x: f32, e0: f32, e1: f32) -> f32
{
    if x < e0
    {
        0.0
    }
    else if x > e1
    {
        1.0
    }
    else
    {
        (x - e0) / (e1 - e0)
    }
}

/// Smooth step function across the range [e0, e1]
/// Denoted as `w` in the original source
fn gt_smoothstep(x: f32, e0: f32, e1: f32) -> f32
{
    if x < e0
    {
        0.0
    }
    else if x > e1
    {
        1.0
    }
    else
    {
        let a: f32 = (x - e0) / (e1 - e0);
        let b: f32 = 3.0 - 2.0 * a;

        a * a * b
    }
}

/// Applies tonemapping to a scalar value.
///
/// Originally designed by Hajime Uchimura for use in *Gran Turismo*.
/// https://www.desmos.com/calculator/gslcdxvipg
/// # Inputs
/// * `p` - maximum brightness
/// * `a` - contrast
/// * `m` - start of linear section
/// * `l` - length of linear section
/// * `c` - black tightness
/// * `b` - minimum brightness
fn gt_tonemap(x: f32, p: f32, a: f32, m: f32, l: f32, c: f32, b: f32) -> f32
{
    debug_assert!(p > 0.0);
    debug_assert!(a > 0.0);
    debug_assert!(m > 0.0 && m < 1.0);
    debug_assert!(l >= 0.0 && l < 1.0);

    if x < 0.0
    {
        b
    }
    else
    {
        let l0: f32 = (p - m) * l / a;

        // Weight of the 'toe' section
        let w0: f32 = 1.0 - gt_smoothstep(x, 0.0, m);
        // Weight of the 'shoulder' section
        let w2: f32 = gt_lerp(x, m + l0, m + l0);
        // Weight of the linear section
        let w1: f32 = 1.0 - w0 - w2;

        let t: f32 = gt_toe(x, m, c, b) * w0;
        let u: f32 = gt_linear(x, a, m) * w1;
        let v: f32 = gt_shoulder(x, p, a, m, l0) * w2;

        t + u + v
    }
}

/// Applies the *Gran Turismo* tonemapping to an RGB value on a per-channel basis
/// # Inputs
/// * `p` - maximum brightness
/// * `a` - contrast
/// * `m` - start of linear section
/// * `l` - length of linear section
/// * `c` - black tightness
/// * `b` - minimum brightness
pub fn gt_tonemap_vector(input: glam::Vec3A, p: f32, a: f32, m: f32, l: f32, c: f32, b: f32) -> glam::Vec3A
{
    let r: f32 = gt_tonemap(input.x, p, a, m, l, c, b);
    let g: f32 = gt_tonemap(input.y, p, a, m, l, c, b);
    let b: f32 = gt_tonemap(input.z, p, a, m, l, c, b);

    glam::Vec3A::new(r, g, b)
}

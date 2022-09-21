// ---------------------
// Tonemapping functions
fn gt_linear(x: vec3<f32>, a: f32, m: f32) -> vec3<f32> { return m + a * (x - m); }

fn gt_toe(x: vec3<f32>, m: f32, c: f32, b: f32) -> vec3<f32> { return m * pow(x / m, vec3<f32>(c)) + b; }

fn gt_shoulder(x: vec3<f32>, p: f32, a: f32, m: f32, l0: f32) -> vec3<f32>
{
    var s0: f32 = m + l0;
    var s1: f32 = m + a * l0;

    var c2: f32 = a * p / (p - s1);

    return p - (p - s1) * exp(-c2 * (x - s0) / p);
}

fn gt_tonemap(x: vec3<f32>, p: f32, a: f32, m: f32, l: f32, c: f32, b: f32) -> vec3<f32>
{
    var l0: f32 = (p - m) * l / a;

    // Weight of the 'toe' section
    var w0: vec3<f32> = 1.0 - smoothstep(vec3<f32>(0.0), vec3<f32>(m), x);
    // Weight of the 'shoulder' section
    var w2: vec3<f32> = step(vec3<f32>(m + l0), x);
    // Weight of the linear section
    var w1: vec3<f32> = 1.0 - w0 - w2;

    var t: vec3<f32> = gt_toe(x, m, c, b) * w0;
    var u: vec3<f32> = gt_linear(x, a, m) * w1;
    var v: vec3<f32> = gt_shoulder(x, p, a, m, l0) * w2;

    return max(t + u + v, vec3<f32>(0.0));
}
// ---------------------

struct PushConstants
{
    num_samples: f32,
}

var<push_constant> push_constants: PushConstants;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) v_index: u32) -> VertexOutput
{
    var uv: vec2<f32> = vec2<f32>(f32((v_index << 1u) & 2u), f32(v_index & 2u));

    var vert_out: VertexOutput;

    vert_out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    vert_out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);

    return vert_out;
}

// Fragment Shader
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

@fragment
fn fs_main(frag_in: VertexOutput) -> @location(0) vec4<f32>
{
    var texture_sample: vec3<f32> = textureSample(t_diffuse, s_diffuse, frag_in.uv).rgb / push_constants.num_samples;

    return vec4<f32>(gt_tonemap(texture_sample, 1.0, 1.0, 0.22, 0.4, 1.33, 0.0), 1.0);
}
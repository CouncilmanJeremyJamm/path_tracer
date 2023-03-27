@group(0) @binding(0) var t_input:        texture_2d<f32>;
@group(0) @binding(1) var t_accumulation: texture_2d<f32>;
@group(0) @binding(2) var t_output:       texture_storage_2d<rgba32float, write>;

@group(1) @binding(0) var t_position:     texture_storage_2d<rgba32float, read>;
@group(1) @binding(1) var t_velocity:     texture_storage_2d<rg32float, read_write>;

@group(2) @binding(0) var t_id:           texture_storage_2d<r32uint, read>;
@group(2) @binding(1) var t_sampler:      sampler;

fn w_divide(v: vec4<f32>) -> vec3<f32>
{
    return v.xyz / max(v.w, 1.0);
}

fn sample_catmull_rom(tex: texture_2d<f32>, tex_size: vec2<f32>, smp: sampler, uv: vec2<f32>) -> vec3<f32>
{
    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.

    let samplePos: vec2<f32> = (uv * tex_size) + 0.5;
    let texPos1: vec2<f32> = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    let f: vec2<f32> = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    let w0: vec2<f32> = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1: vec2<f32> = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2: vec2<f32> = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3: vec2<f32> = f * f * (-0.5 + 0.5 * f);

    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    let w12: vec2<f32> = w1 + w2;
    let offset12: vec2<f32> = w2 / (w1 + w2);

    // Compute the final UV coordinates we'll use for sampling the texture
    let texPos0: vec2<f32> = (texPos1 - 1.0) / tex_size;
    let texPos3: vec2<f32> = (texPos1 + 2.0) / tex_size;
    let texPos12: vec2<f32> = (texPos1 + offset12) / tex_size;

    var c: vec3<f32> = vec3<f32>(0.0);

    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos0.x,  texPos0.y),  0.0)) * w0.x  * w0.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos12.x, texPos0.y),  0.0)) * w12.x * w0.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos3.x,  texPos0.y),  0.0)) * w3.x  * w0.y;

    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos0.x,  texPos12.y), 0.0)) * w0.x  * w12.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos12.x, texPos12.y), 0.0)) * w12.x * w12.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos3.x,  texPos12.y), 0.0)) * w3.x  * w12.y;

    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos0.x,  texPos3.y),  0.0)) * w0.x  * w3.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos12.x, texPos3.y),  0.0)) * w12.x * w3.y;
    c += w_divide(textureSampleLevel(tex, smp, vec2<f32>(texPos3.x,  texPos3.y),  0.0)) * w3.x  * w3.y;

    return c;
}

fn rgb_to_ycocg(in: vec3<f32>) -> vec3<f32>
{
    return mat3x3<f32>(
         0.25, 0.5,  0.25,
         0.5,  0.0, -0.5,
        -0.25, 0.5, -0.25
    ) * in;
}

fn ycocg_to_rgb(in: vec3<f32>) -> vec3<f32>
{
    return mat3x3<f32>(
        1.0,  1.0, -1.0,
        1.0,  0.0,  1.0,
        1.0, -1.0, -1.0
    ) * in;
}

fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, q: vec3<f32>) -> vec3<f32>
{
    // note: only clips towards aabb center (but fast!)
    let p_clip: vec3<f32> = 0.5 * (aabb_max + aabb_min);
    let e_clip: vec3<f32> = 0.5 * (aabb_max - aabb_min);

    let v_clip: vec3<f32> = q - p_clip;
    let v_unit: vec3<f32> = v_clip.xyz / e_clip;
    let a_unit: vec3<f32> = abs(v_unit);
    let ma_unit: f32 = max(a_unit.x, max(a_unit.y, a_unit.z));

    if (ma_unit > 1.0)
    {
        return p_clip + v_clip / ma_unit;
    }
    else
    {
        return q;// point inside aabb
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let current_coords: vec2<i32> = vec2<i32>(global_id.xy);

    // From WGSL spec for textureLoad():
    // If an out-of-bounds access occurs, the built-in function may do any of the following:
    // - not be executed
    // - store `value` to some in bounds texel

//    if (any(current_coords >= textureDimensions(t_velocity)))
//    {
//        return;
//    }

    let current_diffuse: vec3<f32> = textureLoad(t_input, current_coords, 0).xyz;

    let dimensions: vec2<f32> = vec2<f32>(textureDimensions(t_velocity));

    var minimum: vec3<f32> = vec3<f32>(1e10);
    var maximum: vec3<f32> = vec3<f32>(-1e10);

    var m1: vec3<f32> = vec3<f32>(0.0);
    var m2: vec3<f32> = vec3<f32>(0.0);

    var closest_depth: f32 = 1e20;
    var closest_velocity_coords: vec2<i32> = vec2<i32>(0);

    // Calculate bounds of kernel (only use valid coords)
    let min_bounds: vec2<i32> = max(current_coords - 1, vec2<i32>(0));
    let max_bounds: vec2<i32> = min(current_coords + 1, vec2<i32>(dimensions) - 1);

    let n: i32 = (max_bounds.x + 1 - min_bounds.x) * (max_bounds.y + 1 - min_bounds.y);

    // Find min & max for colour clamping
    // TODO: loads current_diffuse twice
    for (var x: i32 = min_bounds.x; x <= max_bounds.x; x++)
    {
        for (var y: i32 = min_bounds.y; y <= max_bounds.y; y++)
        {
            let diffuse_depth: vec4<f32> = textureLoad(t_input, vec2<i32>(x, y), 0);
            let diffuse: vec3<f32> = rgb_to_ycocg(diffuse_depth.rgb);
            let depth: f32 = diffuse_depth.a;

            minimum = min(minimum, diffuse);
            maximum = max(maximum, diffuse);

            m1 += diffuse;
            m2 += diffuse * diffuse;

            if (depth < closest_depth)
            {
                closest_depth = depth;
                closest_velocity_coords = vec2<i32>(x, y);
            }
        }
    }

    let current_uv: vec2<f32> = (vec2<f32>(current_coords) + 0.5) / dimensions;

    let previous_uv: vec2<f32> = current_uv - textureLoad(t_velocity, closest_velocity_coords).xy;
    let previous_coords: vec2<i32> = vec2<i32>(floor(previous_uv * dimensions));

    // Unpack model ids of the current & previous frame
    let current_id: u32 =  textureLoad(t_id, current_coords).x & 0xFFFFu;
    let old_id: u32     = (textureLoad(t_id, previous_coords).x >> 16u) & 0xFFFFu;

    if ((current_id != old_id) || any(previous_coords < vec2<i32>(0)) || any(previous_coords >= vec2<i32>(dimensions)))
    {
        let c0: vec2<f32> = vec2<f32>(current_coords) / dimensions;
        let c1: vec2<f32> = c0 + (vec2<f32>(1.0) / dimensions);

        let a: vec4<f32> = textureSampleLevel(t_input, t_sampler, c0, 0.0);
        let b: vec4<f32> = textureSampleLevel(t_input, t_sampler, vec2<f32>(c0.x, c1.y), 0.0);
        let c: vec4<f32> = textureSampleLevel(t_input, t_sampler, vec2<f32>(c1.x, c0.y), 0.0);
        let d: vec4<f32> = textureSampleLevel(t_input, t_sampler, c1, 0.0);

        textureStore(t_output, current_coords, (a + b + c + d) / 4.0);
    }
    else
    {
        // Set the AABB based on the variance in the 3x3 neighbourhood
        let mu: vec3<f32> = m1 / f32(n);
        let sigma: vec3<f32> = sqrt(m2 / f32(n) - mu * mu);
        let gamma: f32 = 1.0;

        minimum = mu - gamma * sigma;
        maximum = mu + gamma * sigma;

        // Sample the accumulation texture using Catmull-Rom
        let previous_diffuse: vec3<f32> = sample_catmull_rom(
            t_accumulation,
            dimensions,
            t_sampler,
            previous_uv
        );

        // Clamp result from accumulation buffer in YCoCg space,
        // then convert back to RGB
        let clamped_diffuse: vec3<f32> = ycocg_to_rgb(clip_aabb(
            minimum,
            maximum,
            rgb_to_ycocg(previous_diffuse)
        ));

        // Blend between input and accumulation
        let output: vec3<f32> = mix(clamped_diffuse, current_diffuse, 0.15);

        textureStore(t_output, current_coords, vec4<f32>(output, 1.0));
    }
}
@group(0) @binding(0) var t_position: texture_storage_2d<rgba32float, read>;
@group(0) @binding(1) var t_velocity: texture_storage_2d<rg32float, read_write>;

struct PushConstants
{
    last_inv_projection: mat4x4<f32>,
}

var<push_constant> push_constants: PushConstants;

fn w_divide(v: vec4<f32>) -> vec3<f32>
{
    return v.xyz / max(v.w, 1.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let dimensions: vec2<f32> = vec2<f32>(textureDimensions(t_velocity));
    let current_coords: vec2<i32> = vec2<i32>(global_id.xy);

    // From WGSL spec for textureLoad():
    // If an out-of-bounds access occurs, the built-in function may do any of the following:
    // - not be executed
    // - store `value` to some in bounds texel

//    if (any(current_coords >= textureDimensions(t_velocity)))
//    {
//        return;
//    }

    let current_uv: vec2<f32> = (vec2<f32>(current_coords) + 0.5) / dimensions;

    // Get world position
    let position: vec4<f32> = vec4<f32>(textureLoad(t_position, current_coords).xyz, 1.0);
    // Find reprojected coords in the previous frame texture
    let previous_uv: vec2<f32> = w_divide(push_constants.last_inv_projection * position).xy * 0.5 + 0.5;

    textureStore(t_velocity, current_coords, vec4<f32>(current_uv - previous_uv, 0.0, 1.0));
}
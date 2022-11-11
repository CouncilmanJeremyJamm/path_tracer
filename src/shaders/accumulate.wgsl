@group(0) @binding(0) var t_input:        texture_2d<f32>;
@group(0) @binding(1) var t_accumulation: texture_2d<f32>;
@group(0) @binding(2) var t_output:       texture_storage_2d<rgba32float, write>;

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

    // Camera has not moved, add input directly to accumulation texture
    textureStore(t_output, current_coords, textureLoad(t_accumulation, current_coords, 0) + vec4<f32>(current_diffuse, 1.0));
}
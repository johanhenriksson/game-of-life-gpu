#version 450

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(binding = 0) uniform sampler2D image;

vec3 empty = vec3(0);

vec3 getGradient(vec4 c1, vec4 c2, vec4 c3, vec4 c4, float value_);

vec3 magic(float v) {
    return getGradient(
        vec4(0.349, 0.757, 0.451, 0),
        vec4(0.631, 0.498, 0.878, 0.33),
        vec4(0.365, 0.149, 0.757, 0.67),
        vec4(0.349, 0.757, 0.451, 1),
        v
    );
}

vec3 magic2(float v) {
    return getGradient(
        vec4(1.000, 0.435, 0.569, 0),
        vec4(0.518, 0.369, 0.761, 0.33),
        vec4(0.365, 0.149, 0.757, 0.67),
        vec4(1.000, 0.588, 0.443, 1),
        v
    );
}

vec3 magic3(float v) {
    return getGradient(
        vec4(0.310, 0.110, 0.941, 0),
        vec4(0.490, 0.988, 0.247, 0.33),
        vec4(0.988, 0.384, 0.227, 0.67),
        vec4(0.310, 0.110, 0.941, 1),
        v
    );
}

void main() {
    vec4 cell = texture(image, v_uv);

    // pick color based on generation
    vec3 grad = magic2(cell.z);
    vec3 color = grad;

    if (cell.x <= 0) {
        // dead
        if (cell.y > 0.01) { 
            // decaying
            color = mix(empty, color, cell.y);
        } else {
            // empty space 
            color = empty;
        }
    } else {
        // alive
    }

    float gamma = 1;
    f_color = vec4(pow(color.rgb, vec3(gamma)), 1);
}


/// @brief  calculates a gradient mapped colour 
/// @detail the calculations are based on colour stops c1, c2, c3, 
///         and one normlised input value, emphasises constant 
///         execution time.
/// @param c1 [rgbw] first colour stop, with .w being the normalised 
///         position of the colour stop on the gradient beam.
/// @param c2 [rgbw] second colour stop, with .w being the normalised 
///         position of the colour stop on the gradient beam.
/// @param c3 [rgbw] third colour stop, with .w being the normalised 
///         position of the colour stop on the gradient beam.
/// @param c4 [rgbw] fourth colour stop, with .w being the normalised 
///         position of the colour stop on the gradient beam.
/// @param value the input value for gradient mapping, a normalised 
///        float
/// @note   values are interpolated close to sinusoidal, using 
///         smoothstep, sinusoidal interpolation being what Photoshop
///         uses for its gradients.
/// @author @tgfrerer
vec3 getGradient(vec4 c1, vec4 c2, vec4 c3, vec4 c4, float value_){
	
	float blend1 = smoothstep(c1.w, c2.w, value_);
	float blend2 = smoothstep(c2.w, c3.w, value_);
	float blend3 = smoothstep(c3.w, c4.w, value_);
	
	vec3 
	col = mix(c1.rgb, c2.rgb, blend1);
	col = mix(col, c3.rgb, blend2);
	col = mix(col, c4.rgb, blend3);
	
	return col;
}

#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba8, set = 0, binding = 0) writeonly uniform image2D next;
layout (rgba8, set = 0, binding = 1) readonly uniform image2D prev;

void main() {
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(next);

    // vec4 topColor = vec4(1,0,0,1);
    // vec4 bottomColor = vec4(0,0,1,1);

    if(texelCoord.x < size.x && texelCoord.y < size.y) {
        vec4 current = imageLoad(prev, texelCoord);

        current += vec4(0.01, 0.0, 0.0, 1.0);
        if (current.x > 1.0) {
            current.x = 0.0;
        }

        // float blend = float(texelCoord.y) / (size.y); 
        imageStore(next, texelCoord, current); //mix(topColor, bottomColor, blend));
    }
}

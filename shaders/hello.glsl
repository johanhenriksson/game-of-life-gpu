#version 450

layout (local_size_x = 16, local_size_y = 16) in;

layout (rgba16f, set = 0, binding = 0) uniform image2D next;

void main() {
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(next);

    vec4 topColor = vec4(1,0,0,1);
    vec4 bottomColor = vec4(0,0,1,1);

    if(texelCoord.x < size.x && texelCoord.y < size.y) {
        float blend = float(texelCoord.y) / (size.y); 
        imageStore(next, texelCoord, mix(topColor, bottomColor, blend));
    }
}

#version 450

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(binding = 0) uniform sampler2D image;

void main() {
    f_color = texture(image, v_uv) + vec4(v_uv,0,1);
}

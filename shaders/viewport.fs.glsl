#version 450

layout(location = 0) in vec2 v_uv;

layout(location = 0) out vec4 f_color;

layout(binding = 0) uniform sampler2D image;

void main() {
    vec2 cell = texture(image, v_uv).xy;

    if (cell.x > 0.02 || cell.y > 0.02) {
        f_color = vec4(cell.x, cell.y, cell.y, 1);
    } else {
        discard;
    }
}

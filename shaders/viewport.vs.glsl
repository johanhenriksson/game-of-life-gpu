#version 450

layout(location = 0) out vec2 v_uv;

vec2 vertices[4] = vec2[4](
    vec2(0, 0), // top left 0
    vec2(0, 1), // top right 1
    vec2(1, 0), // bottom left 2
    vec2(1, 1) // bottom right 3
);

vec2 uvs[4] = vec2[4](
    vec2(0, 0), // top left 0
    vec2(0, 1), // top right 1
    vec2(1, 0), // bottom left 2
    vec2(1, 1) // bottom right 3
);

int indices[6] = int[6](
    0, 3, 2,
    3, 0, 1
);

layout(push_constant) uniform PushConstants {
    mat4 proj;
    mat4 model;
} push;

void main() {
    int idx = indices[gl_VertexIndex];
    vec2 pos = vertices[idx];

    gl_Position = push.proj * push.model * vec4(pos, 0, 1);
    v_uv = uvs[idx];
}

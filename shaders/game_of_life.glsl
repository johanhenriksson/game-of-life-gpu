#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba8, set = 0, binding = 0) writeonly uniform image2D next;
layout (rgba8, set = 0, binding = 1) readonly uniform image2D prev;

layout(push_constant) uniform PushConstants {
    int enabled;
} pushConstants;

void main() {
    ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(next);

    vec4 self = imageLoad(prev, cell);
    bool was_alive = self.x > 0.0;

    int neighbors = 0;
    if (was_alive) {
        neighbors = -1;
    }
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            ivec2 neighbor = cell + ivec2(x, y);
            if(neighbor.x >= 0 && neighbor.x < size.x && neighbor.y >= 0 && neighbor.y < size.y) {
                vec4 current = imageLoad(prev, neighbor);
                if(current.x > 0.0) {
                    neighbors++;
                }
            }
        }
    }

    bool alive = was_alive;
    if (pushConstants.enabled > 0) {
        // execute logic
        if (was_alive) {
            if (neighbors < 2 || neighbors > 3) {
                alive = false;
            }
        } else {
            if (neighbors == 3) {
                alive = true;
            }
        } 
    }

    vec4 color = vec4(0);
    if (alive) {
        color = vec4(1);
    }
    imageStore(next, cell, color); 
}

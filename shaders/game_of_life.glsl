#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba8, set = 0, binding = 0) writeonly uniform image2D next;
layout (rgba8, set = 0, binding = 1) readonly uniform image2D prev;

layout(push_constant) uniform PushConstants {
    int enabled;
} pushConstants;

bool cell_alive(vec4 cell) {
    return cell.x > 0.5;
}

void main() {
    ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(next);

    vec4 color = imageLoad(prev, cell);
    bool was_alive = cell_alive(color);

    int neighbors = 0;
    if (was_alive) {
        neighbors = -1;
    }
    for(int y = -1; y <= 1; y++) {
        for(int x = -1; x <= 1; x++) {
            ivec2 neighbor = cell + ivec2(x, y);

            // wrap around edges
            if(neighbor.x < 0) neighbor.x += size.x;
            if(neighbor.x >= size.x) neighbor.x -= size.x;
            if(neighbor.y < 0) neighbor.y += size.y;
            if(neighbor.y >= size.y) neighbor.y -= size.y;

            vec4 current = imageLoad(prev, neighbor);
            if(cell_alive(current)) {
                neighbors++;
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

    color.x = 0;
    color.y *= 0.9;
    color.z *= 0.9;
    if (color.y < 0.01) {
        color.y = 0;
    }
    if (color.z < 0.01) {
        color.z = 0;
    }

    if (alive) {
        color=vec4(1);
    }
    imageStore(next, cell, color); 
}

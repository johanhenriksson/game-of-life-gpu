#version 450

layout (local_size_x = 32, local_size_y = 32) in;

layout (rgba8, set = 0, binding = 0) writeonly uniform image2D next;
layout (rgba8, set = 0, binding = 1) readonly uniform image2D prev;

layout(push_constant) uniform PushConstants {
    int enabled;
    int generation;
} push;

bool cell_alive(vec4 cell) {
    return cell.x > 0.5;
}

void main() {
    ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = imageSize(next);
    vec4 color = imageLoad(prev, cell);

    if (push.enabled == 0) {
        imageStore(next, cell, color); 
        return;
    }

    bool was_alive = cell_alive(color);

    // count neighbors
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

    // compute next iteration
    bool alive = was_alive;
    if (was_alive) {
        if (neighbors < 2 || neighbors > 3) {
            // died
            alive = false;
            color.x = 0;
            color.y = 0.8;
        }
    } else {
        if (neighbors == 3) {
            // spawned
            alive = true;
            color.x = 1;
            color.z = push.generation / 255.0;
        }
    } 

    // compute pixel color based on alive state
    if (alive) {
    } else {
        // decay
        color.y *= 0.9;
    }

    imageStore(next, cell, color); 
}

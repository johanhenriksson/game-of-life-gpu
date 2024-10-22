const sdl = @import("sdl2");
const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");

const ComputePipe = @import("compute_pipe.zig").ComputePipe;
const ComputeArgs = @import("compute_pipe.zig").PushConstants;
const GraphicsPipe = @import("graphics_pipe.zig").GraphicsPipe;
const GraphicsArgs = @import("graphics_pipe.zig").GraphicsArgs;
const VkContext = @import("context.zig").VkContext;
const Swapchain = @import("swapchain.zig").Swapchain;

const Cursor = @import("cursor.zig").Cursor;
const CursorView = @import("cursor.zig").CursorView;
const Viewport = @import("viewport.zig").Viewport;
const WorldView = @import("world.zig").WorldView;
const Pattern = @import("pattern.zig").Pattern;
const Library = @import("pattern.zig").Library;

const Vec2 = @import("primitives.zig").Vec2;

const app_name = "zulkan";

pub fn main() !void {
    try sdl.init(.{
        .video = true,
        .events = true,
    });
    defer sdl.quit();

    var extent = vk.Extent2D{ .width = 1200, .height = 800 };

    var window = try sdl.createWindow(
        app_name,
        .{ .centered = {} },
        .{ .centered = {} },
        extent.width,
        extent.height,
        .{
            .vis = .shown,
            .context = .vulkan,
            .resizable = true,
        },
    );
    defer window.destroy();

    const allocator = std.heap.page_allocator;

    const ctx = try VkContext.init(allocator, app_name, window);
    defer ctx.deinit();

    std.debug.print("Using device: {?s}\n", .{ctx.props.device_name});

    var swapchain = try Swapchain.init(&ctx, allocator, extent);
    defer swapchain.deinit();

    std.debug.print("Created swapchain with {d} images\n", .{swapchain.swap_images.len});

    const pool = try ctx.vkd.createCommandPool(ctx.dev, &.{
        .flags = .{},
        .queue_family_index = ctx.graphics_queue.family,
    }, null);
    defer ctx.vkd.destroyCommandPool(ctx.dev, pool, null);

    var compute = try ComputePipe.init(&ctx, allocator, pool, vk.Extent2D{
        .width = 1000,
        .height = 1000,
    });
    defer compute.deinit();

    var graphics = try GraphicsPipe.init(&ctx, &swapchain, allocator);
    defer graphics.deinit();

    var world = try WorldView.init(
        &ctx,
        allocator,
        &graphics,
        &compute,
    );
    defer world.deinit();

    var viewport = try Viewport.init(
        &ctx,
        @floatFromInt(extent.width),
        @floatFromInt(extent.height),
        @floatFromInt(compute.extent.width / 2),
        @floatFromInt(compute.extent.height / 2),
    );
    defer viewport.deinit();

    var library = try Library.loadDir(allocator, "cells");
    const cursor_size = @max(library.max_width, library.max_height);
    std.debug.print("Loaded {d} patterns\n", .{library.patterns.len});

    std.debug.print("Creating cursor with size {d}x{d}\n", .{ cursor_size, cursor_size });
    var cursor = try Cursor.init(&ctx, pool, cursor_size);
    defer cursor.deinit();

    library.random();
    try cursor.setPattern(library.current());

    var cursor_view = try CursorView.init(&ctx, allocator, &cursor, &graphics);
    defer cursor_view.deinit();

    var cmdbufs = [3]vk.CommandBuffer{ .null_handle, .null_handle, .null_handle };

    const interval = 1e9 / 20;
    var lastTick = try std.time.Instant.now();
    var placeHeld = false;
    var panHeld = false;
    var mouse = Vec2{ .x = 0, .y = 0 };

    var running = true;
    var simulating = false;
    while (running) {
        const frame = swapchain.image_index;
        const prev_frame = (frame + swapchain.swap_images.len - 1) % swapchain.swap_images.len;

        while (sdl.pollEvent()) |event| {
            switch (event) {
                sdl.Event.quit => {
                    running = false;
                },
                sdl.Event.key_down => {
                    switch (event.key_down.keycode) {
                        .escape => { // quit
                            running = false;
                        },
                        .space => { // pause
                            simulating = !simulating;
                        },
                        .c => { // clear
                            try compute.clear(pool);
                        },

                        // pan
                        .a => viewport.pan(-10, 0),
                        .d => viewport.pan(10, 0),
                        .w => viewport.pan(0, -10),
                        .s => viewport.pan(0, 10),

                        .r => { // random cursor
                            library.random();
                            try cursor.setPattern(library.current());
                        },
                        .@"1" => { // select first pattern
                            try library.select(0);
                            try cursor.setPattern(library.current());
                        },
                        .n => {
                            library.next();
                            try cursor.setPattern(library.current());
                        },
                        .p => {
                            library.prev();
                            try cursor.setPattern(library.current());
                        },

                        else => {},
                    }
                },
                sdl.Event.mouse_button_down => |down| {
                    mouse = Vec2{ .x = @floatFromInt(down.x), .y = @floatFromInt(down.y) };
                    switch (down.button) {
                        .right => {
                            panHeld = true;
                        },
                        .left => {
                            placeHeld = true;
                            const w = viewport.screenToWorld(Vec2{ .x = @floatFromInt(down.x), .y = @floatFromInt(down.y) });
                            try cursor.paste(pool, compute.buffers[prev_frame], @intFromFloat(w.x), @intFromFloat(w.y));
                        },
                        else => {},
                    }
                },
                sdl.Event.mouse_button_up => |up| {
                    switch (up.button) {
                        .right => {
                            panHeld = false;
                        },
                        .left => {
                            placeHeld = false;
                        },
                        else => {},
                    }
                },
                sdl.Event.mouse_motion => |move| {
                    mouse = Vec2{ .x = @floatFromInt(move.x), .y = @floatFromInt(move.y) };
                    if (panHeld) {
                        viewport.pan(@floatFromInt(-move.delta_x), @floatFromInt(-move.delta_y));
                    }
                },
                sdl.Event.mouse_wheel => |wheel| {
                    viewport.zoom(@floatFromInt(wheel.delta_y));
                },
                else => {},
            }
        }

        const mouse_world = viewport.screenToWorld(mouse);
        cursor.setPosition(mouse_world);

        if (placeHeld) {
            // const w = viewport.screenToWorld(Vec2{ .x = mouse.x, .y = mouse.y });
            // try cursor.paste(pool, compute.buffers[prev_frame], @intFromFloat(w.x), @intFromFloat(w.y));
        }

        const now = try std.time.Instant.now();
        const elapsed = now.since(lastTick);
        var tick = false;
        if (simulating and elapsed > interval) {
            lastTick = now;
            tick = true;
        }

        // TODO: rewrite this
        // awkward way to avoid releasing command buffers that are still in use

        // wait for the current frame to finish rendering
        // so that we can re-record its command buffer
        try swapchain.waitForRender();

        if (cmdbufs[frame] != .null_handle) {
            ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&cmdbufs[frame]));
            cmdbufs[frame] = .null_handle;
        }
        cmdbufs[frame] = try createCommandBuffers(
            &ctx,
            &viewport,
            &graphics,
            &compute,
            pool,
            &world,
            &cursor_view,
            frame,
            tick,
        );
        const cmdbuf = cmdbufs[frame];

        //
        // present frame
        //

        const state = swapchain.present(cmdbuf) catch |err| switch (err) {
            error.OutOfDateKHR => Swapchain.PresentState.suboptimal,
            else => |narrow| return narrow,
        };

        if (state == .suboptimal) {
            const size = window.getSize();
            extent.width = @intCast(size.width);
            extent.height = @intCast(size.height);
            std.debug.print("(swap) Resizing to: {d}x{d}\n", .{ extent.width, extent.height });
            try swapchain.recreate(extent);
            try graphics.resize(&swapchain);
            viewport.resize(@floatFromInt(extent.width), @floatFromInt(extent.height));

            std.debug.print("(swap) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
        }
    }
    std.debug.print("Exiting...\n", .{});

    try swapchain.waitForAllFences();
}

fn createCommandBuffers(
    ctx: *const VkContext,
    view: *const Viewport,
    gfx: *const GraphicsPipe,
    compute: *const ComputePipe,
    pool: vk.CommandPool,
    world: *WorldView,
    cursor: *CursorView,
    frame: usize,
    tick: bool,
) !vk.CommandBuffer {
    var cmdbuf: vk.CommandBuffer = undefined;

    try ctx.vkd.allocateCommandBuffers(ctx.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    errdefer ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&cmdbuf));

    try ctx.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

    //
    // execute compute shader
    //

    compute.execute(cmdbuf, frame, tick);

    // add a memory barrier to ensure the compute shader is finished before the graphics pipeline starts
    // compute shader (write) must finish before the fragment shader runs (read)
    ctx.vkd.cmdPipelineBarrier(cmdbuf, .{ .compute_shader_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, &[_]vk.ImageMemoryBarrier{
        .{
            .image = compute.buffers[frame],
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = .general,
            .new_layout = .general,
            .src_queue_family_index = ctx.graphics_queue.family,
            .dst_queue_family_index = ctx.graphics_queue.family,
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        },
    });

    //
    // draw the graphics
    //

    gfx.begin(cmdbuf, frame);

    // draw the world
    world.draw(cmdbuf, view.matrix(), frame);

    // draw the cursor
    cursor.draw(cmdbuf, view.matrix(), frame);

    gfx.end(cmdbuf);

    try ctx.vkd.endCommandBuffer(cmdbuf);

    return cmdbuf;
}

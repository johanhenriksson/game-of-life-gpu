const std = @import("std");
const sdl = @import("sdl2");
const vk = @import("vulkan");
const Vertex = @import("graphics_pipe.zig").Vertex;
const ComputePipe = @import("compute_pipe.zig").ComputePipe;
const ComputeArgs = @import("compute_pipe.zig").PushConstants;
const GraphicsPipe = @import("graphics_pipe.zig").GraphicsPipe;
const GraphicsArgs = @import("graphics_pipe.zig").GraphicsArgs;
const VkContext = @import("context.zig").VkContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;
const Viewport = @import("viewport.zig").Viewport;

const zm = @import("zmath");
const Vec2 = @import("primitives.zig").Vec2;
const Rect = @import("primitives.zig").Rect;
const Color = @import("primitives.zig").Color;
const transitionImages = @import("helper.zig").transitionImages;
const copyBuffer = @import("helper.zig").copyBuffer;

const Cursor = @import("cursor.zig").Cursor;

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

    var compute = try ComputePipe.init(&ctx, allocator, vk.Extent2D{
        .width = 1000,
        .height = 1000,
    });
    defer compute.deinit();

    var graphics = try GraphicsPipe.init(&ctx, &swapchain, allocator);
    defer graphics.deinit();

    const pool = try ctx.vkd.createCommandPool(ctx.dev, &.{
        .flags = .{},
        .queue_family_index = ctx.graphics_queue.family,
    }, null);
    defer ctx.vkd.destroyCommandPool(ctx.dev, pool, null);

    var viewport = try Viewport.init(
        &ctx,
        @floatFromInt(extent.width),
        @floatFromInt(extent.height),
        @floatFromInt(compute.extent.width / 2),
        @floatFromInt(compute.extent.height / 2),
    );
    defer viewport.deinit();

    var cursor = try Cursor.init(&ctx, pool, 3, 3);
    createGlider(&cursor);
    defer cursor.deinit();

    try transitionImages(&ctx, pool, compute.buffers, vk.ImageLayout.undefined, vk.ImageLayout.general);

    for (0..swapchain.swap_images.len) |frame| {
        ctx.vkd.updateDescriptorSets(ctx.dev, 1, &[_]vk.WriteDescriptorSet{
            .{
                .dst_set = graphics.descriptors[frame],
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = vk.DescriptorType.combined_image_sampler,
                .p_image_info = &[_]vk.DescriptorImageInfo{
                    .{
                        .sampler = compute.buffer_sampler[frame],
                        .image_view = compute.buffer_views[frame],
                        .image_layout = vk.ImageLayout.general,
                    },
                },
                .p_buffer_info = &[_]vk.DescriptorBufferInfo{},
                .p_texel_buffer_view = &[_]vk.BufferView{},
            },
        }, 0, null);
    }

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
                sdl.Event.key_down => {
                    if (event.key_down.keycode == sdl.Keycode.escape) {
                        running = false;
                    }
                    if (event.key_down.keycode == sdl.Keycode.space) {
                        simulating = !simulating;
                    }
                    if (event.key_down.keycode == sdl.Keycode.a) {
                        viewport.pan(-10, 0);
                    }
                    if (event.key_down.keycode == sdl.Keycode.d) {
                        viewport.pan(10, 0);
                    }
                    if (event.key_down.keycode == sdl.Keycode.w) {
                        viewport.pan(0, -10);
                    }
                    if (event.key_down.keycode == sdl.Keycode.s) {
                        viewport.pan(0, 10);
                    }
                },
                sdl.Event.quit => {
                    running = false;
                },
                sdl.Event.window => |window_event| switch (window_event.type) {
                    .resized => |resize_event| {
                        const width: u32 = @intCast(resize_event.width);
                        const height: u32 = @intCast(resize_event.height);
                        if ((extent.width != width) or (extent.height != height)) {
                            extent.width = width;
                            extent.height = height;
                            std.debug.print("(sdl) Resizing to: {d}x{d}\n", .{ extent.width, extent.height });
                            try swapchain.recreate(extent);
                            try graphics.resize(&swapchain);
                            viewport.resize(@floatFromInt(extent.width), @floatFromInt(extent.height));

                            std.debug.print("(sdl) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
                        }
                    },
                    else => {},
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

        if (placeHeld) {
            const w = viewport.screenToWorld(Vec2{ .x = mouse.x, .y = mouse.y });
            try cursor.paste(pool, compute.buffers[prev_frame], @intFromFloat(w.x), @intFromFloat(w.y));
        }

        const now = try std.time.Instant.now();
        const elapsed = now.since(lastTick);
        var tick = false;
        if (simulating and elapsed > interval) {
            lastTick = now;
            tick = true;
        }

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
            swapchain.extent,
            frame,
            tick,
        );
        const cmdbuf = cmdbufs[frame];

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

        // std.time.sleep(1e8);
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
    extent: vk.Extent2D,
    i: usize,
    tick: bool,
) !vk.CommandBuffer {
    var cmdbuf: vk.CommandBuffer = undefined;

    try ctx.vkd.allocateCommandBuffers(ctx.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    errdefer ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&cmdbuf));

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0.05, 0.05, 0.05, 1 } },
    };

    const viewport = vk.Viewport{
        .x = 0,
        .y = 0,
        .width = @as(f32, @floatFromInt(extent.width)),
        .height = @as(f32, @floatFromInt(extent.height)),
        .min_depth = 0,
        .max_depth = 1,
    };

    const scissor = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    try ctx.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{},
        .p_inheritance_info = null,
    });

    //
    // execute compute shader
    //

    ctx.vkd.cmdPushConstants(cmdbuf, compute.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(ComputeArgs), @ptrCast(&ComputeArgs{
        .enabled = if (tick) 1 else 0,
    }));

    ctx.vkd.cmdBindDescriptorSets(cmdbuf, .compute, compute.pipeline_layout, 0, 1, @ptrCast(&compute.descriptors[i]), 0, null);
    ctx.vkd.cmdBindPipeline(cmdbuf, .compute, compute.pipeline);
    ctx.vkd.cmdDispatch(cmdbuf, compute.extent.width, compute.extent.height, 1);

    // add a memory barrier to ensure the compute shader is finished before the graphics pipeline starts
    // compute shader (write) must finish before the fragment shader runs (read)
    ctx.vkd.cmdPipelineBarrier(cmdbuf, .{ .compute_shader_bit = true }, .{ .fragment_shader_bit = true }, .{}, 0, null, 0, null, 1, &[_]vk.ImageMemoryBarrier{
        .{
            .image = compute.buffers[i],
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
            .old_layout = vk.ImageLayout.general,
            .new_layout = vk.ImageLayout.general,
            .src_queue_family_index = ctx.graphics_queue.family,
            .dst_queue_family_index = ctx.graphics_queue.family,
            .subresource_range = vk.ImageSubresourceRange{
                .aspect_mask = vk.ImageAspectFlags{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        },
    });

    //
    // draw the graphics pipeline
    //

    ctx.vkd.cmdSetViewport(cmdbuf, 0, 1, @as([*]const vk.Viewport, @ptrCast(&viewport)));
    ctx.vkd.cmdSetScissor(cmdbuf, 0, 1, @as([*]const vk.Rect2D, @ptrCast(&scissor)));

    // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
    const render_area = vk.Rect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = extent,
    };

    ctx.vkd.cmdBeginRenderPass(cmdbuf, &.{
        .render_pass = gfx.render_pass,
        .framebuffer = gfx.framebuffers[i],
        .render_area = render_area,
        .clear_value_count = 1,
        .p_clear_values = @as([*]const vk.ClearValue, @ptrCast(&clear)),
    }, .@"inline");

    ctx.vkd.cmdPushConstants(cmdbuf, gfx.pipeline_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(GraphicsArgs), @ptrCast(&GraphicsArgs{
        .proj = view.matrix(),
        .model = zm.scaling(@floatFromInt(compute.extent.width), @floatFromInt(compute.extent.height), 1),
        .size = Vec2{
            .x = @floatFromInt(compute.extent.width),
            .y = @floatFromInt(compute.extent.height),
        },
    }));

    ctx.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, gfx.pipeline_layout, 0, 1, @as([*]const vk.DescriptorSet, @ptrCast(&gfx.descriptors[i])), 0, null);
    ctx.vkd.cmdBindPipeline(cmdbuf, .graphics, gfx.pipeline);
    ctx.vkd.cmdDraw(cmdbuf, 6, 1, 0, 0);

    ctx.vkd.cmdEndRenderPass(cmdbuf);
    try ctx.vkd.endCommandBuffer(cmdbuf);

    return cmdbuf;
}

fn createGlider(cursor: *const Cursor) void {
    const alive = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };

    cursor.set(0, 0, alive);
    cursor.set(0, 1, alive);
    cursor.set(0, 2, alive);
    cursor.set(1, 0, alive);
    cursor.set(1, 1, Color{ .r = 0, .g = 0, .b = 0, .a = 255 });
    cursor.set(1, 2, alive);
    cursor.set(2, 0, alive);
    cursor.set(2, 1, alive);
    cursor.set(2, 2, alive);
}

const std = @import("std");
const sdl = @import("sdl2");
const vk = @import("vulkan");
const Vertex = @import("graphics_pipe.zig").Vertex;
const ComputePipe = @import("compute_pipe.zig").ComputePipe;
const GraphicsPipe = @import("graphics_pipe.zig").GraphicsPipe;
const VkContext = @import("context.zig").VkContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Allocator = std.mem.Allocator;

const app_name = "zulkan";

const vertices = [_]Vertex{
    .{ .pos = .{ -1, -1 }, .uv = .{ 0, 0 } }, // top left
    .{ .pos = .{ 1, 1 }, .uv = .{ 1, 1 } }, // bottom right
    .{ .pos = .{ -1, 1 }, .uv = .{ 1, 0 } }, // bottom left
    .{ .pos = .{ 1, 1 }, .uv = .{ 1, 1 } }, // bottom right
    .{ .pos = .{ -1, -1 }, .uv = .{ 0, 0 } }, // top left
    .{ .pos = .{ 1, -1 }, .uv = .{ 0, 1 } }, // top right
};

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
        .width = 32,
        .height = 32,
    });
    defer compute.deinit();

    var graphics = try GraphicsPipe.init(&ctx, &swapchain, allocator);
    defer graphics.deinit();

    const pool = try ctx.vkd.createCommandPool(ctx.dev, &.{
        .flags = .{},
        .queue_family_index = ctx.graphics_queue.family,
    }, null);
    defer ctx.vkd.destroyCommandPool(ctx.dev, pool, null);

    const buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
        .flags = .{},
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer ctx.vkd.destroyBuffer(ctx.dev, buffer, null);
    const mem_reqs = ctx.vkd.getBufferMemoryRequirements(ctx.dev, buffer);
    const memory = try ctx.allocate(mem_reqs, .{ .device_local_bit = true });
    defer ctx.vkd.freeMemory(ctx.dev, memory, null);
    try ctx.vkd.bindBufferMemory(ctx.dev, buffer, memory, 0);

    try uploadVertices(&ctx, pool, buffer);

    std.debug.print("Uploaded vertices\n", .{});

    var cmdbufs = try createCommandBuffers(
        &ctx,
        pool,
        allocator,
        buffer,
        swapchain.extent,
        &graphics,
    );
    defer destroyCommandBuffers(&ctx, pool, allocator, cmdbufs);

    var running = true;
    while (running) {
        while (sdl.pollEvent()) |event| {
            switch (event) {
                sdl.Event.key_down => {
                    if (event.key_down.keycode == sdl.Keycode.escape) {
                        running = false;
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

                            destroyCommandBuffers(&ctx, pool, allocator, cmdbufs);
                            cmdbufs = try createCommandBuffers(
                                &ctx,
                                pool,
                                allocator,
                                buffer,
                                swapchain.extent,
                                &graphics,
                            );
                            std.debug.print("(sdl) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
                        }
                    },
                    else => {},
                },
                else => {},
            }
        }
        const cmdbuf = cmdbufs[swapchain.image_index];

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

            destroyCommandBuffers(&ctx, pool, allocator, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &ctx,
                pool,
                allocator,
                buffer,
                swapchain.extent,
                &graphics,
            );

            std.debug.print("(swap) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
        }
    }
    std.debug.print("Exiting...\n", .{});

    try swapchain.waitForAllFences();
}

fn uploadVertices(gc: *const VkContext, pool: vk.CommandPool, buffer: vk.Buffer) !void {
    const staging_buffer = try gc.vkd.createBuffer(gc.dev, &.{
        .flags = .{},
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer gc.vkd.destroyBuffer(gc.dev, staging_buffer, null);
    const mem_reqs = gc.vkd.getBufferMemoryRequirements(gc.dev, staging_buffer);
    const staging_memory = try gc.allocate(mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer gc.vkd.freeMemory(gc.dev, staging_memory, null);
    try gc.vkd.bindBufferMemory(gc.dev, staging_buffer, staging_memory, 0);

    {
        const data = try gc.vkd.mapMemory(gc.dev, staging_memory, 0, vk.WHOLE_SIZE, .{});
        defer gc.vkd.unmapMemory(gc.dev, staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        for (vertices, 0..) |vertex, i| {
            gpu_vertices[i] = vertex;
        }
    }

    try copyBuffer(gc, pool, buffer, staging_buffer, @sizeOf(@TypeOf(vertices)));
}

fn copyBuffer(gc: *const VkContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
    var cmdbuf: vk.CommandBuffer = undefined;
    try gc.vkd.allocateCommandBuffers(gc.dev, &.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    defer gc.vkd.freeCommandBuffers(gc.dev, pool, 1, @ptrCast(&cmdbuf));

    try gc.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });

    const region = vk.BufferCopy{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    gc.vkd.cmdCopyBuffer(cmdbuf, src, dst, 1, @ptrCast(&region));

    try gc.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try gc.vkd.queueSubmit(gc.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try gc.vkd.queueWaitIdle(gc.graphics_queue.handle);
}

fn createCommandBuffers(
    ctx: *const VkContext,
    pool: vk.CommandPool,
    allocator: Allocator,
    buffer: vk.Buffer,
    extent: vk.Extent2D,
    gfx: *GraphicsPipe,
    // images: []vk.ImageView,
) ![]vk.CommandBuffer {
    const cmdbufs = try allocator.alloc(vk.CommandBuffer, gfx.framebuffers.len);
    errdefer allocator.free(cmdbufs);

    try ctx.vkd.allocateCommandBuffers(ctx.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = @truncate(cmdbufs.len),
    }, cmdbufs.ptr);
    errdefer ctx.vkd.freeCommandBuffers(ctx.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);

    const clear = vk.ClearValue{
        .color = .{ .float_32 = .{ 0, 0, 0, 1 } },
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

    for (cmdbufs, 0..) |cmdbuf, i| {
        try ctx.vkd.beginCommandBuffer(cmdbuf, &.{
            .flags = .{},
            .p_inheritance_info = null,
        });

        // execute compute shader

        // add a memory barrier to ensure the compute shader is finished before the graphics pipeline starts

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

        ctx.vkd.cmdBindPipeline(cmdbuf, .graphics, gfx.pipeline);
        const offset = [_]vk.DeviceSize{0};
        ctx.vkd.cmdBindVertexBuffers(cmdbuf, 0, 1, @as([*]const vk.Buffer, @ptrCast(&buffer)), &offset);
        ctx.vkd.cmdDraw(cmdbuf, vertices.len, 1, 0, 0);

        ctx.vkd.cmdEndRenderPass(cmdbuf);
        try ctx.vkd.endCommandBuffer(cmdbuf);
    }

    return cmdbufs;
}

fn destroyCommandBuffers(gc: *const VkContext, pool: vk.CommandPool, allocator: Allocator, cmdbufs: []vk.CommandBuffer) void {
    gc.vkd.freeCommandBuffers(gc.dev, pool, @truncate(cmdbufs.len), cmdbufs.ptr);
    allocator.free(cmdbufs);
}

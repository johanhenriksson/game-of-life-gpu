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

    const test_image_size = compute.extent.width * compute.extent.height * @sizeOf(Color);
    std.debug.print("Source buffer size {}\n", .{test_image_size});
    const test_buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
        .flags = .{},
        .size = test_image_size,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer ctx.vkd.destroyBuffer(ctx.dev, test_buffer, null);
    const test_mem_reqs = ctx.vkd.getBufferMemoryRequirements(ctx.dev, test_buffer);
    const test_memory = try ctx.allocate(test_mem_reqs, .{ .host_visible_bit = true });
    defer ctx.vkd.freeMemory(ctx.dev, test_memory, null);
    try ctx.vkd.bindBufferMemory(ctx.dev, test_buffer, test_memory, 0);

    const test_image_ptr = try ctx.vkd.mapMemory(ctx.dev, test_memory, 0, vk.WHOLE_SIZE, .{});
    defer ctx.vkd.unmapMemory(ctx.dev, test_memory);

    const test_image_data = try createTestImage(allocator, compute.extent);
    const test_image_colors: [*]Color = @ptrCast(@alignCast(test_image_ptr));
    for (test_image_data, 0..) |color, i| {
        test_image_colors[i] = color;
    }

    var imgbuf: vk.CommandBuffer = .null_handle;
    try ctx.vkd.allocateCommandBuffers(ctx.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&imgbuf));
    defer ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&imgbuf));
    try ctx.vkd.beginCommandBuffer(imgbuf, &vk.CommandBufferBeginInfo{});
    for (0..compute.buffers.len) |i| {
        ctx.vkd.cmdPipelineBarrier(imgbuf, .{ .all_commands_bit = true }, .{ .all_commands_bit = true }, .{}, 0, null, 0, null, 1, &[_]vk.ImageMemoryBarrier{
            .{
                .image = compute.buffers[i],
                .src_access_mask = .{ .memory_read_bit = true, .memory_write_bit = true },
                .dst_access_mask = .{ .memory_read_bit = true, .memory_write_bit = true },
                .old_layout = vk.ImageLayout.undefined,
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
    }
    try ctx.vkd.endCommandBuffer(imgbuf);
    try ctx.vkd.queueSubmit(ctx.graphics_queue.handle, 1, &[_]vk.SubmitInfo{
        .{
            .wait_semaphore_count = 0,
            .p_wait_semaphores = null,
            .p_wait_dst_stage_mask = null,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&imgbuf),
            .signal_semaphore_count = 0,
            .p_signal_semaphores = null,
        },
    }, .null_handle);
    try ctx.vkd.queueWaitIdle(ctx.graphics_queue.handle);

    for (0..compute.buffers.len) |i| {
        const target_buf = try ctx.vkd.createBuffer(ctx.dev, &vk.BufferCreateInfo{
            .size = test_image_size,
            .usage = .{
                .transfer_dst_bit = true,
            },
            .sharing_mode = vk.SharingMode.exclusive,
        }, null);
        try ctx.vkd.bindBufferMemory(ctx.dev, target_buf, compute.buffer_memory[i], 0);
        try copyBuffer(&ctx, pool, target_buf, test_buffer, compute.extent.width * compute.extent.height * @sizeOf(Color));
        ctx.vkd.destroyBuffer(ctx.dev, target_buf, null);
    }

    std.debug.print("Uploaded test image\n", .{});

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
    compute.next_image(0);

    var cmdbufs = try createCommandBuffers(
        &ctx,
        pool,
        allocator,
        buffer,
        swapchain.extent,
        &graphics,
        &compute,
    );
    defer destroyCommandBuffers(&ctx, pool, allocator, cmdbufs);
    std.debug.print("Created command buffers\n", .{});

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
                                &compute,
                            );
                            std.debug.print("(sdl) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
                        }
                    },
                    else => {},
                },
                else => {},
            }
        }

        const frame = swapchain.image_index;
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
        std.debug.print("updated graphics[{}]\n", .{frame});

        compute.next_image(frame);
        std.debug.print("updated compute[{}]\n", .{frame});

        const cmdbuf = cmdbufs[frame];

        std.debug.print("submit frame {d}\n", .{frame});
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
                &compute,
            );

            std.debug.print("(swap) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
        }
    }
    std.debug.print("Exiting...\n", .{});

    try swapchain.waitForAllFences();
}

fn uploadVertices(ctx: *const VkContext, pool: vk.CommandPool, buffer: vk.Buffer) !void {
    const staging_buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
        .flags = .{},
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer ctx.vkd.destroyBuffer(ctx.dev, staging_buffer, null);
    const mem_reqs = ctx.vkd.getBufferMemoryRequirements(ctx.dev, staging_buffer);
    const staging_memory = try ctx.allocate(mem_reqs, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer ctx.vkd.freeMemory(ctx.dev, staging_memory, null);
    try ctx.vkd.bindBufferMemory(ctx.dev, staging_buffer, staging_memory, 0);

    {
        const data = try ctx.vkd.mapMemory(ctx.dev, staging_memory, 0, vk.WHOLE_SIZE, .{});
        defer ctx.vkd.unmapMemory(ctx.dev, staging_memory);

        const gpu_vertices: [*]Vertex = @ptrCast(@alignCast(data));
        for (vertices, 0..) |vertex, i| {
            gpu_vertices[i] = vertex;
        }
    }

    try copyBuffer(ctx, pool, buffer, staging_buffer, @sizeOf(@TypeOf(vertices)));
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
    compute: *ComputePipe,
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

        ctx.vkd.cmdBindDescriptorSets(cmdbuf, .compute, compute.pipeline_layout, 0, 1, compute.descriptors.ptr, 0, null);
        ctx.vkd.cmdBindPipeline(cmdbuf, .compute, compute.pipeline);
        ctx.vkd.cmdDispatch(cmdbuf, 16, 16, 1);

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

        ctx.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, gfx.pipeline_layout, 0, 1, @as([*]const vk.DescriptorSet, @ptrCast(&gfx.descriptors[i])), 0, null);
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

const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

fn createTestImage(allocator: std.mem.Allocator, extent: vk.Extent2D) ![]Color {
    const size = extent.width * extent.height;
    const img = try allocator.alloc(Color, size);
    for (0..extent.height) |y| {
        for (0..extent.width) |x| {
            img[y * extent.width + x] = Color{
                .r = 255,
                .g = 0,
                .b = 0,
                .a = 255,
            };
        }
    }
    return img;
}

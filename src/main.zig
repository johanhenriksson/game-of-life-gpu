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

    const vertex_buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
        .flags = .{},
        .size = @sizeOf(@TypeOf(vertices)),
        .usage = .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer ctx.vkd.destroyBuffer(ctx.dev, vertex_buffer, null);
    const vertex_mem_reqs = ctx.vkd.getBufferMemoryRequirements(ctx.dev, vertex_buffer);
    const vertex_memory = try ctx.allocate(vertex_mem_reqs, .{ .device_local_bit = true });
    defer ctx.vkd.freeMemory(ctx.dev, vertex_memory, null);
    try ctx.vkd.bindBufferMemory(ctx.dev, vertex_buffer, vertex_memory, 0);

    try uploadVertices(&ctx, pool, vertex_buffer);

    const cursor_image = try ctx.vkd.createImage(ctx.dev, &.{
        .flags = .{},
        .image_type = vk.ImageType.@"2d",
        .format = vk.Format.r8g8b8a8_unorm,
        .extent = vk.Extent3D{ .width = 32, .height = 32, .depth = 1 },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = vk.ImageTiling.linear,
        .usage = .{ .transfer_src_bit = true, .sampled_bit = true },
        .sharing_mode = vk.SharingMode.exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
        .initial_layout = vk.ImageLayout.preinitialized,
    }, null);
    defer ctx.vkd.destroyImage(ctx.dev, cursor_image, null);
    const cursor_mem_req = ctx.vkd.getImageMemoryRequirements(ctx.dev, cursor_image);
    const cursor_memory = try ctx.allocate(cursor_mem_req, .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer ctx.vkd.freeMemory(ctx.dev, cursor_memory, null);
    try ctx.vkd.bindImageMemory(ctx.dev, cursor_image, cursor_memory, 0);

    const cursor_buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
        .flags = .{},
        .size = cursor_mem_req.size,
        .usage = .{ .transfer_src_bit = true },
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
    }, null);
    defer ctx.vkd.destroyBuffer(ctx.dev, cursor_buffer, null);
    try ctx.vkd.bindBufferMemory(ctx.dev, cursor_buffer, cursor_memory, 0);

    const test_image_ptr = try ctx.vkd.mapMemory(ctx.dev, cursor_memory, 0, vk.WHOLE_SIZE, .{});
    defer ctx.vkd.unmapMemory(ctx.dev, cursor_memory);
    const test_image_data = try createTestImage(allocator, compute.extent);
    const test_image_colors: [*]Color = @ptrCast(@alignCast(test_image_ptr));
    for (test_image_data, 0..) |color, i| {
        test_image_colors[i] = color;
    }

    try transitionImage(&ctx, pool, cursor_image, vk.ImageLayout.undefined, vk.ImageLayout.general);
    try transitionImages(&ctx, pool, compute.buffers, vk.ImageLayout.undefined, vk.ImageLayout.general);

    // copy cursor to compute initial buffer (which is the last frame)
    const src_region = [2]vk.Offset2D{ .{ .x = 0, .y = 0 }, .{ .x = 32, .y = 32 } };
    const dst_region = [2]vk.Offset2D{ .{ .x = 0, .y = 0 }, .{ .x = 32, .y = 32 } };
    try blitImage(&ctx, pool, cursor_image, compute.buffers[compute.buffers.len - 1], src_region, dst_region);

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

    var cmdbufs = try createCommandBuffers(
        &ctx,
        pool,
        allocator,
        vertex_buffer,
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
                                vertex_buffer,
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

            destroyCommandBuffers(&ctx, pool, allocator, cmdbufs);
            cmdbufs = try createCommandBuffers(
                &ctx,
                pool,
                allocator,
                vertex_buffer,
                swapchain.extent,
                &graphics,
                &compute,
            );

            std.debug.print("(swap) Resized to: {d}x{d}\n", .{ extent.width, extent.height });
        }

        std.time.sleep(1e9);
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

fn blitImage(ctx: *const VkContext, pool: vk.CommandPool, src: vk.Image, dst: vk.Image, src_region: [2]vk.Offset2D, dst_region: [2]vk.Offset2D) !void {
    var cmdbuf: vk.CommandBuffer = undefined;
    try ctx.vkd.allocateCommandBuffers(ctx.dev, &.{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmdbuf));
    defer ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&cmdbuf));

    try ctx.vkd.beginCommandBuffer(cmdbuf, &.{
        .flags = .{ .one_time_submit_bit = true },
        .p_inheritance_info = null,
    });

    ctx.vkd.cmdBlitImage(
        cmdbuf,
        src,
        vk.ImageLayout.general,
        dst,
        vk.ImageLayout.general,
        1,
        &[_]vk.ImageBlit{
            .{
                .src_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .src_offsets = [2]vk.Offset3D{
                    .{ .x = src_region[0].x, .y = src_region[0].y, .z = 0 },
                    .{ .x = src_region[1].x, .y = src_region[1].y, .z = 1 },
                },
                .dst_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .dst_offsets = [2]vk.Offset3D{
                    .{ .x = dst_region[0].x, .y = dst_region[0].y, .z = 0 }, .{ .x = dst_region[1].x, .y = dst_region[1].y, .z = 1 },
                },
            },
        },
        vk.Filter.nearest,
    );

    try ctx.vkd.endCommandBuffer(cmdbuf);

    const si = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmdbuf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try ctx.vkd.queueSubmit(ctx.graphics_queue.handle, 1, @ptrCast(&si), .null_handle);
    try ctx.vkd.queueWaitIdle(ctx.graphics_queue.handle);
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

        ctx.vkd.cmdBindDescriptorSets(cmdbuf, .compute, compute.pipeline_layout, 0, 1, @ptrCast(&compute.descriptors[i]), 0, null);
        ctx.vkd.cmdBindPipeline(cmdbuf, .compute, compute.pipeline);
        ctx.vkd.cmdDispatch(cmdbuf, 16, 16, 1);

        // add a memory barrier to ensure the compute shader is finished before the graphics pipeline starts
        // compute shader (write) must finish before the fragment shader runs (read)
        ctx.vkd.cmdPipelineBarrier(
            cmdbuf,
            .{ .compute_shader_bit = true },
            .{ .fragment_shader_bit = true },
            .{},
            0,
            null,
            0,
            null,
            1,
            &[_]vk.ImageMemoryBarrier{
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
            },
        );

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
                .r = 255 * std.crypto.random.intRangeAtMost(u8, 0, 1),
                .g = 0,
                .b = 0,
                .a = 255,
            };
        }
    }
    // top-left is always white
    img[0] = Color{ .r = 255, .g = 255, .b = 255, .a = 255 };
    return img;
}

fn transitionImage(ctx: *const VkContext, pool: vk.CommandPool, image: vk.Image, from_layout: vk.ImageLayout, to_layout: vk.ImageLayout) !void {
    var images = [_]vk.Image{image};
    return try transitionImages(ctx, pool, &images, from_layout, to_layout);
}

fn transitionImages(ctx: *const VkContext, pool: vk.CommandPool, images: []vk.Image, from_layout: vk.ImageLayout, to_layout: vk.ImageLayout) !void {
    var imgbuf: vk.CommandBuffer = .null_handle;
    try ctx.vkd.allocateCommandBuffers(ctx.dev, &vk.CommandBufferAllocateInfo{
        .command_pool = pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&imgbuf));
    defer ctx.vkd.freeCommandBuffers(ctx.dev, pool, 1, @ptrCast(&imgbuf));
    try ctx.vkd.beginCommandBuffer(imgbuf, &vk.CommandBufferBeginInfo{});
    for (images) |image| {
        ctx.vkd.cmdPipelineBarrier(imgbuf, .{ .all_commands_bit = true }, .{ .all_commands_bit = true }, .{}, 0, null, 0, null, 1, &[_]vk.ImageMemoryBarrier{
            .{
                .image = image,
                .src_access_mask = .{ .memory_read_bit = true, .memory_write_bit = true },
                .dst_access_mask = .{ .memory_read_bit = true, .memory_write_bit = true },
                .old_layout = from_layout,
                .new_layout = to_layout,
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
}

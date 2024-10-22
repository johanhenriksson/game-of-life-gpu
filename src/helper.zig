const Color = @import("primitives.zig").Color;
const Rect = @import("primitives.zig").Rect;
const VkContext = @import("context.zig").VkContext;
const vk = @import("vulkan");

pub fn blitImage(ctx: *const VkContext, pool: vk.CommandPool, src: vk.Image, dst: vk.Image, src_rect: Rect, dst_rect: Rect) !void {
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
                .src_offsets = src_rect.offsets3D(),
                .dst_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .dst_offsets = dst_rect.offsets3D(),
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

pub fn transitionImage(ctx: *const VkContext, pool: vk.CommandPool, image: vk.Image, from_layout: vk.ImageLayout, to_layout: vk.ImageLayout) !void {
    var images = [_]vk.Image{image};
    return try transitionImages(ctx, pool, &images, from_layout, to_layout);
}

pub fn transitionImages(ctx: *const VkContext, pool: vk.CommandPool, images: []vk.Image, from_layout: vk.ImageLayout, to_layout: vk.ImageLayout) !void {
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

pub fn copyBuffer(gc: *const VkContext, pool: vk.CommandPool, dst: vk.Buffer, src: vk.Buffer, size: vk.DeviceSize) !void {
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

pub fn clearImage(ctx: *const VkContext, pool: vk.CommandPool, image: vk.Image, color: Color) !void {
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

    // Define clear color value
    const clear_color = vk.ClearColorValue{
        .float_32 = .{
            @as(f32, @floatFromInt(color.r)) / 255.0,
            @as(f32, @floatFromInt(color.g)) / 255.0,
            @as(f32, @floatFromInt(color.b)) / 255.0,
            @as(f32, @floatFromInt(color.a)) / 255.0,
        },
    };

    // Define range to clear
    const range = vk.ImageSubresourceRange{
        .aspect_mask = .{ .color_bit = true },
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };

    // Clear the image
    ctx.vkd.cmdClearColorImage(
        cmdbuf,
        image,
        .general, // Make sure image is in general layout
        &clear_color,
        1,
        @ptrCast(&range),
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

const std = @import("std");
const vk = @import("vulkan");

const VkContext = @import("context.zig").VkContext;
const Shader = @import("shader.zig");

const transitionImages = @import("helper.zig").transitionImages;

pub const ComputeArgs = struct {
    enabled: i32,
};

pub const ComputePipe = struct {
    ctx: *const VkContext,
    allocator: std.mem.Allocator,

    module: vk.ShaderModule,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,

    descriptor_pool: vk.DescriptorPool,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptors: []vk.DescriptorSet,

    extent: vk.Extent2D,
    buffers: []vk.Image,
    buffer_views: []vk.ImageView,
    buffer_memory: []vk.DeviceMemory,

    pub fn init(ctx: *const VkContext, allocator: std.mem.Allocator, pool: vk.CommandPool, extent: vk.Extent2D) !ComputePipe {
        const shader = try Shader.compile(ctx, allocator, Shader.Stage.compute, "shaders/game_of_life.glsl");
        const frames = 3;
        const format = vk.Format.r8g8_unorm;

        const descriptor_pool = try ctx.vkd.createDescriptorPool(ctx.dev, &vk.DescriptorPoolCreateInfo{
            .max_sets = @intCast(frames),
            .pool_size_count = 1,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{
                .{
                    .type = vk.DescriptorType.storage_image,
                    .descriptor_count = 100,
                },
            },
        }, null);

        const layout = try ctx.vkd.createDescriptorSetLayout(ctx.dev, &vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 2,
            .p_bindings = &[_]vk.DescriptorSetLayoutBinding{
                .{
                    .binding = 0,
                    .descriptor_type = vk.DescriptorType.storage_image,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .compute_bit = true,
                    },
                },
                .{
                    .binding = 1,
                    .descriptor_type = vk.DescriptorType.storage_image,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .compute_bit = true,
                    },
                },
            },
        }, null);

        const descriptors = try allocator.alloc(vk.DescriptorSet, frames);
        try ctx.vkd.allocateDescriptorSets(ctx.dev, &vk.DescriptorSetAllocateInfo{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = @intCast(descriptors.len),
            .p_set_layouts = &[_]vk.DescriptorSetLayout{ layout, layout, layout },
        }, descriptors.ptr);

        const pushConstantRange = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = @sizeOf(ComputeArgs),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.dev, &vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{layout},

            .push_constant_range_count = 1,
            .p_push_constant_ranges = &[_]vk.PushConstantRange{pushConstantRange},
        }, null);
        errdefer ctx.vkd.destroyDescriptorSetLayout(ctx.dev, layout, null);

        var pipeline = vk.Pipeline.null_handle;
        const result = try ctx.vkd.createComputePipelines(ctx.dev, vk.PipelineCache.null_handle, 1, &[_]vk.ComputePipelineCreateInfo{
            .{
                .layout = pipeline_layout,
                .stage = .{
                    .p_name = "main",
                    .module = shader,
                    .stage = .{
                        .compute_bit = true,
                    },
                },
                .base_pipeline_handle = vk.Pipeline.null_handle,
                .base_pipeline_index = 0,
            },
        }, null, @ptrCast(&pipeline));
        if (result != vk.Result.success) {
            return error.@"Failed to create compute pipeline";
        }
        errdefer ctx.vkd.destroyShaderModule(ctx.dev, shader, null);

        const buffers = try allocator.alloc(vk.Image, frames);
        const buffer_memory = try allocator.alloc(vk.DeviceMemory, frames);
        const buffer_views = try allocator.alloc(vk.ImageView, frames);
        for (0..buffers.len) |i| {
            buffers[i] = try ctx.vkd.createImage(ctx.dev, &vk.ImageCreateInfo{
                .image_type = vk.ImageType.@"2d",
                .format = format,
                .extent = .{
                    .width = extent.width,
                    .height = extent.height,
                    .depth = 1,
                },
                .mip_levels = 1,
                .array_layers = 1,
                .samples = .{
                    .@"1_bit" = true,
                },
                .tiling = vk.ImageTiling.optimal,
                .usage = .{
                    .sampled_bit = true,
                    .transfer_dst_bit = true,
                    .transfer_src_bit = true,
                    .storage_bit = true,
                },
                .sharing_mode = vk.SharingMode.exclusive,
                .initial_layout = vk.ImageLayout.undefined,
            }, null);
            errdefer ctx.vkd.destroyImage(ctx.dev, buffers[i], null);

            const memreq = ctx.vkd.getImageMemoryRequirements(ctx.dev, buffers[i]);
            buffer_memory[i] = try ctx.allocate(memreq, vk.MemoryPropertyFlags{
                .device_local_bit = true,
            });

            try ctx.vkd.bindImageMemory(ctx.dev, buffers[i], buffer_memory[i], 0);

            buffer_views[i] = try ctx.vkd.createImageView(ctx.dev, &vk.ImageViewCreateInfo{
                .image = buffers[i],
                .view_type = vk.ImageViewType.@"2d",
                .format = format,
                .components = .{
                    .r = vk.ComponentSwizzle.identity,
                    .g = vk.ComponentSwizzle.identity,
                    .b = vk.ComponentSwizzle.identity,
                    .a = vk.ComponentSwizzle.identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{
                        .color_bit = true,
                    },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            }, null);
        }

        try transitionImages(ctx, pool, buffers, vk.ImageLayout.undefined, vk.ImageLayout.general);

        for (0..frames) |frame| {
            const prev = if (frame > 0) frame - 1 else buffers.len - 1;

            ctx.vkd.updateDescriptorSets(ctx.dev, 2, &[_]vk.WriteDescriptorSet{
                .{
                    .dst_set = descriptors[frame],
                    .dst_binding = 0,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_image,
                    .p_image_info = &[_]vk.DescriptorImageInfo{
                        .{
                            .sampler = .null_handle,
                            .image_view = buffer_views[frame],
                            .image_layout = .general,
                        },
                    },
                    .p_buffer_info = &[_]vk.DescriptorBufferInfo{},
                    .p_texel_buffer_view = &[_]vk.BufferView{},
                },
                .{
                    .dst_set = descriptors[frame],
                    .dst_binding = 1,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .storage_image,
                    .p_image_info = &[_]vk.DescriptorImageInfo{
                        .{
                            .sampler = .null_handle,
                            .image_view = buffer_views[prev],
                            .image_layout = .general,
                        },
                    },
                    .p_buffer_info = &[_]vk.DescriptorBufferInfo{},
                    .p_texel_buffer_view = &[_]vk.BufferView{},
                },
            }, 0, null);
        }

        return ComputePipe{
            .ctx = ctx,
            .allocator = allocator,

            .module = shader,
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .descriptor_pool = descriptor_pool,
            .descriptor_set_layout = layout,
            .descriptors = descriptors,

            .extent = extent,
            .buffers = buffers,
            .buffer_memory = buffer_memory,
            .buffer_views = buffer_views,
        };
    }

    pub fn deinit(self: *ComputePipe) void {
        self.ctx.vkd.destroyPipeline(self.ctx.dev, self.pipeline, null);
        self.ctx.vkd.destroyPipelineLayout(self.ctx.dev, self.pipeline_layout, null);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.dev, self.descriptor_pool, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.dev, self.descriptor_set_layout, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.module, null);

        for (0..self.buffers.len) |i| {
            self.ctx.vkd.destroyImageView(self.ctx.dev, self.buffer_views[i], null);
            self.ctx.vkd.destroyImage(self.ctx.dev, self.buffers[i], null);
            self.ctx.vkd.freeMemory(self.ctx.dev, self.buffer_memory[i], null);
        }

        self.allocator.free(self.buffer_views);
        self.allocator.free(self.buffers);
        self.allocator.free(self.buffer_memory);
    }

    pub fn execute(self: *const ComputePipe, cmdbuf: vk.CommandBuffer, frame: usize, tick: bool) void {
        self.ctx.vkd.cmdPushConstants(cmdbuf, self.pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(ComputeArgs), @ptrCast(&ComputeArgs{
            .enabled = if (tick) 1 else 0,
        }));

        self.ctx.vkd.cmdBindDescriptorSets(cmdbuf, .compute, self.pipeline_layout, 0, 1, @ptrCast(&self.descriptors[frame]), 0, null);
        self.ctx.vkd.cmdBindPipeline(cmdbuf, .compute, self.pipeline);
        self.ctx.vkd.cmdDispatch(cmdbuf, self.extent.width, self.extent.height, 1);
    }
};

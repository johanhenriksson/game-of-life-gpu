const std = @import("std");
const vk = @import("vulkan");
const VkContext = @import("context.zig").VkContext;
const Shader = @import("shader.zig");

pub const ComputePipe = struct {
    ctx: *const VkContext,
    allocator: std.mem.Allocator,

    module: vk.ShaderModule,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,

    descriptor_pool: vk.DescriptorPool,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptors: vk.DescriptorSet,

    extent: vk.Extent2D,
    buffers: []vk.Image,
    buffer_views: []vk.ImageView,
    buffer_memory: []vk.DeviceMemory,
    buffer_sampler: []vk.Sampler,

    pub fn init(ctx: *const VkContext, allocator: std.mem.Allocator, extent: vk.Extent2D) !ComputePipe {
        const shader = try Shader.compile(ctx, allocator, Shader.Stage.compute, "shaders/hello.glsl");

        const pool = try ctx.vkd.createDescriptorPool(ctx.dev, &vk.DescriptorPoolCreateInfo{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{
                .{
                    .type = vk.DescriptorType.sampler,
                    .descriptor_count = 100,
                },
            },
        }, null);

        const layout = try ctx.vkd.createDescriptorSetLayout(ctx.dev, &vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1,
            .p_bindings = &[_]vk.DescriptorSetLayoutBinding{
                .{
                    .binding = 0,
                    .descriptor_type = vk.DescriptorType.sampler,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .compute_bit = true,
                    },
                },
            },
        }, null);

        var descriptor: vk.DescriptorSet = .null_handle;
        try ctx.vkd.allocateDescriptorSets(ctx.dev, &vk.DescriptorSetAllocateInfo{
            .descriptor_pool = pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{layout},
        }, @ptrCast(&descriptor));

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.dev, &vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{layout},
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

        const frames = 3;
        const buffers = try allocator.alloc(vk.Image, frames);
        const buffer_memory = try allocator.alloc(vk.DeviceMemory, frames);
        const buffer_views = try allocator.alloc(vk.ImageView, frames);
        const buffer_sampler = try allocator.alloc(vk.Sampler, frames);
        for (0..buffers.len) |i| {
            buffers[i] = try ctx.vkd.createImage(ctx.dev, &vk.ImageCreateInfo{
                .image_type = vk.ImageType.@"2d",
                .format = vk.Format.r8g8b8a8_unorm,
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
                },
                .sharing_mode = vk.SharingMode.exclusive,
                .initial_layout = vk.ImageLayout.general,
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
                .format = vk.Format.r8g8b8a8_unorm,
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

            buffer_sampler[i] = try ctx.vkd.createSampler(ctx.dev, &vk.SamplerCreateInfo{
                .mag_filter = vk.Filter.nearest,
                .min_filter = vk.Filter.nearest,
                .mipmap_mode = vk.SamplerMipmapMode.nearest,
                .address_mode_u = vk.SamplerAddressMode.clamp_to_edge,
                .address_mode_v = vk.SamplerAddressMode.clamp_to_edge,
                .address_mode_w = vk.SamplerAddressMode.clamp_to_edge,
                .mip_lod_bias = 0,
                .anisotropy_enable = 0,
                .max_anisotropy = 0,
                .compare_enable = 0,
                .compare_op = vk.CompareOp.less,
                .min_lod = 0,
                .max_lod = 0,
                .border_color = vk.BorderColor.float_opaque_black,
                .unnormalized_coordinates = 0,
            }, null);
        }

        return ComputePipe{
            .ctx = ctx,
            .allocator = allocator,

            .module = shader,
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .descriptor_pool = pool,
            .descriptor_set_layout = layout,
            .descriptors = descriptor,

            .extent = extent,
            .buffers = buffers,
            .buffer_memory = buffer_memory,
            .buffer_views = buffer_views,
            .buffer_sampler = buffer_sampler,
        };
    }

    pub fn deinit(self: *ComputePipe) void {
        self.ctx.vkd.destroyPipeline(self.ctx.dev, self.pipeline, null);
        self.ctx.vkd.destroyPipelineLayout(self.ctx.dev, self.pipeline_layout, null);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.dev, self.descriptor_pool, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.dev, self.descriptor_set_layout, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.module, null);

        for (0..self.buffers.len) |i| {
            self.ctx.vkd.destroySampler(self.ctx.dev, self.buffer_sampler[i], null);
            self.ctx.vkd.destroyImageView(self.ctx.dev, self.buffer_views[i], null);
            self.ctx.vkd.destroyImage(self.ctx.dev, self.buffers[i], null);
            self.ctx.vkd.freeMemory(self.ctx.dev, self.buffer_memory[i], null);
        }

        self.allocator.free(self.buffer_sampler);
        self.allocator.free(self.buffer_views);
        self.allocator.free(self.buffers);
        self.allocator.free(self.buffer_memory);
    }

    pub fn next_image(self: *ComputePipe, frame: usize) void {
        const prev = if (frame > 0) frame - 1 else self.buffers.len - 1;
        self.ctx.vkd.updateDescriptorSets(self.ctx.dev, 1, &[_]vk.WriteDescriptorSet{
            .{
                .dst_set = self.descriptors,
                .dst_binding = 0,
                .dst_array_element = 0,
                .descriptor_count = 1,
                .descriptor_type = vk.DescriptorType.sampler,
                .p_image_info = &[_]vk.DescriptorImageInfo{
                    .{
                        .sampler = self.buffer_sampler[frame],
                        .image_view = self.buffer_views[frame],
                        .image_layout = vk.ImageLayout.general,
                    },
                    .{
                        .sampler = self.buffer_sampler[prev],
                        .image_view = self.buffer_views[prev],
                        .image_layout = vk.ImageLayout.general,
                    },
                },
                .p_buffer_info = &[_]vk.DescriptorBufferInfo{},
                .p_texel_buffer_view = &[_]vk.BufferView{},
            },
        }, 0, null);
    }
};

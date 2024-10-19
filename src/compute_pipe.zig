const std = @import("std");
const vk = @import("vulkan");
const VkContext = @import("context.zig").VkContext;
const Shader = @import("shader.zig");

pub const ComputePipe = struct {
    ctx: *const VkContext,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,

    descriptor_pool: vk.DescriptorPool,
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptors: vk.DescriptorSet,

    extent: vk.Extent2D,
    buffers: [2]vk.Image,

    pub fn init(ctx: *const VkContext, allocator: std.mem.Allocator, extent: vk.Extent2D) !ComputePipe {
        const shader = try Shader.compile(ctx, allocator, Shader.Stage.compute, "shaders/hello.glsl");

        const pool = try ctx.vkd.createDescriptorPool(ctx.dev, &vk.DescriptorPoolCreateInfo{
            .max_sets = 1,
            .pool_size_count = 1,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{
                .{
                    .type = vk.DescriptorType.sampler,
                    .descriptor_count = 1,
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

        var buffers = [2]vk.Image{ .null_handle, .null_handle };
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
                .initial_layout = vk.ImageLayout.undefined,
            }, null);
            errdefer ctx.vkd.destroyImage(ctx.dev, buffers[i], null);
        }

        return ComputePipe{
            .ctx = ctx,
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .descriptor_pool = pool,
            .descriptor_set_layout = layout,
            .descriptors = descriptor,

            .extent = extent,
            .buffers = buffers,
        };
    }

    pub fn deinit(self: *ComputePipe) void {
        self.ctx.vkd.destroyPipeline(self.ctx.dev, self.pipeline, null);
        self.ctx.vkd.destroyPipelineLayout(self.ctx.dev, self.pipeline_layout, null);
        self.ctx.vkd.destroyDescriptorPool(self.ctx.dev, self.descriptor_pool, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.dev, self.descriptor_set_layout, null);

        for (0..self.buffers.len) |i| {
            self.ctx.vkd.destroyImage(self.ctx.dev, self.buffers[i], null);
        }
    }
};

const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");

const Color = @import("primitives.zig").Color;
const Rect = @import("primitives.zig").Rect;
const Vec2 = @import("primitives.zig").Vec2;
const Pattern = @import("pattern.zig").Pattern;

const VkContext = @import("context.zig").VkContext;
const Shader = @import("shader.zig");
const ComputePipe = @import("compute_pipe.zig").ComputePipe;
const GraphicsPipe = @import("graphics_pipe.zig").GraphicsPipe;
const GraphicsArgs = @import("graphics_pipe.zig").GraphicsArgs;
const createPipeline = @import("graphics_pipe.zig").createPipeline;

const transitionImage = @import("helper.zig").transitionImage;
const blitImage = @import("helper.zig").blitImage;

pub const World = struct {};

pub const WorldView = struct {
    ctx: *const VkContext,
    gfx: *const GraphicsPipe,
    compute: *const ComputePipe,

    pipeline: vk.Pipeline,
    shader: vk.ShaderModule,
    sampler: vk.Sampler,
    descriptors: []vk.DescriptorSet,

    pub fn init(ctx: *const VkContext, allocator: std.mem.Allocator, gfx: *GraphicsPipe, compute: *ComputePipe) !WorldView {
        const sampler = try ctx.vkd.createSampler(ctx.dev, &.{
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .mipmap_mode = .nearest,
            .address_mode_u = .clamp_to_edge,
            .address_mode_v = .clamp_to_edge,
            .address_mode_w = .clamp_to_edge,
            .mip_lod_bias = 0,
            .anisotropy_enable = vk.TRUE,
            .max_anisotropy = 0,
            .compare_enable = vk.FALSE,
            .compare_op = .always,
            .min_lod = 0,
            .max_lod = 0,
            .border_color = .float_transparent_black,
            .unnormalized_coordinates = vk.FALSE,
        }, null);

        // allocate descriptor sets
        const descriptors = try allocator.alloc(vk.DescriptorSet, compute.buffers.len);
        try ctx.vkd.allocateDescriptorSets(ctx.dev, &.{
            .descriptor_pool = gfx.descriptor_pool,
            .descriptor_set_count = @intCast(descriptors.len),
            .p_set_layouts = &[_]vk.DescriptorSetLayout{ gfx.descriptor_layout, gfx.descriptor_layout, gfx.descriptor_layout },
        }, descriptors.ptr);

        // update descriptor values
        for (0..descriptors.len) |i| {
            ctx.vkd.updateDescriptorSets(ctx.dev, 1, &[_]vk.WriteDescriptorSet{
                .{
                    .dst_set = descriptors[i],
                    .dst_binding = 0,
                    .dst_array_element = 0,
                    .descriptor_count = 1,
                    .descriptor_type = .combined_image_sampler,
                    .p_image_info = &[_]vk.DescriptorImageInfo{
                        .{
                            .sampler = sampler,
                            .image_view = compute.buffer_views[i],
                            .image_layout = .shader_read_only_optimal,
                        },
                    },
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
            }, 0, null);
        }

        const shader = try Shader.compile(ctx, allocator, Shader.Stage.fragment, "shaders/world.fs.glsl");

        const pipeline = try createPipeline(ctx, gfx.pipeline_layout, gfx.render_pass, gfx.vertex, shader);

        return WorldView{
            .ctx = ctx,
            .gfx = gfx,
            .compute = compute,

            .sampler = sampler,
            .shader = shader,
            .pipeline = pipeline,
            .descriptors = descriptors,
        };
    }

    pub fn deinit(self: *const WorldView) void {
        self.ctx.vkd.destroyPipeline(self.ctx.dev, self.pipeline, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.shader, null);

        self.ctx.vkd.destroySampler(self.ctx.dev, self.sampler, null);
        // todo: free descriptors
        // todo: free descriptor array
    }

    pub fn draw(self: *WorldView, cmdbuf: vk.CommandBuffer, view: zm.Mat, frame: usize) void {
        const scaling = zm.scaling(@floatFromInt(self.compute.extent.width), @floatFromInt(self.compute.extent.height), 1);
        const model = scaling;

        self.ctx.vkd.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);

        self.ctx.vkd.cmdPushConstants(cmdbuf, self.gfx.pipeline_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(GraphicsArgs), @ptrCast(&GraphicsArgs{
            .proj = view,
            .model = model,
        }));
        self.ctx.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, self.gfx.pipeline_layout, 0, 1, @as([*]const vk.DescriptorSet, @ptrCast(&self.descriptors[frame])), 0, null);
        self.ctx.vkd.cmdDraw(cmdbuf, 6, 1, 0, 0);
    }
};

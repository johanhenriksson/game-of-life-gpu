const std = @import("std");
const vk = @import("vulkan");
const VkContext = @import("context.zig").VkContext;
const Swapchain = @import("swapchain.zig").Swapchain;
const Shader = @import("shader.zig");
const Vec2 = @import("primitives.zig").Vec2;

const zm = @import("zmath");

pub const GraphicsArgs = struct {
    proj: zm.Mat,
    model: zm.Mat,
    size: Vec2,
    tex_scale: Vec2,
};

pub const GraphicsPipe = struct {
    ctx: *const VkContext,
    allocator: std.mem.Allocator,

    pipeline_layout: vk.PipelineLayout,
    descriptor_pool: vk.DescriptorPool,
    descriptor_layout: vk.DescriptorSetLayout,
    descriptors: []vk.DescriptorSet,

    extent: vk.Extent2D,
    framebuffers: []vk.Framebuffer,
    render_pass: vk.RenderPass,
    vertex: vk.ShaderModule,
    fragment: vk.ShaderModule,

    pub fn init(ctx: *const VkContext, swapchain: *Swapchain, allocator: std.mem.Allocator) !GraphicsPipe {
        const frames = swapchain.swap_images.len;
        const vertex = try Shader.compile(ctx, allocator, Shader.Stage.vertex, "shaders/rect.vs.glsl");
        const fragment = try Shader.compile(ctx, allocator, Shader.Stage.fragment, "shaders/world.fs.glsl");

        const pool = try ctx.vkd.createDescriptorPool(ctx.dev, &vk.DescriptorPoolCreateInfo{
            .max_sets = 1000,
            .pool_size_count = 1,
            .p_pool_sizes = &[_]vk.DescriptorPoolSize{
                .{
                    .type = .combined_image_sampler,
                    .descriptor_count = 1000,
                },
            },
        }, null);

        const layout = try ctx.vkd.createDescriptorSetLayout(ctx.dev, &vk.DescriptorSetLayoutCreateInfo{
            .binding_count = 1,
            .p_bindings = &[_]vk.DescriptorSetLayoutBinding{
                .{
                    .binding = 0,
                    .descriptor_type = .combined_image_sampler,
                    .descriptor_count = 1,
                    .stage_flags = .{
                        .fragment_bit = true,
                    },
                },
            },
        }, null);

        const descriptors = try allocator.alloc(vk.DescriptorSet, frames);
        const layouts = try allocator.alloc(vk.DescriptorSetLayout, frames);
        defer allocator.free(layouts);
        for (0..layouts.len) |i| {
            layouts[i] = layout;
        }
        try ctx.vkd.allocateDescriptorSets(ctx.dev, &.{
            .descriptor_pool = pool,
            .descriptor_set_count = @intCast(descriptors.len),
            .p_set_layouts = layouts.ptr,
        }, descriptors.ptr);

        const pushConstantRange = vk.PushConstantRange{
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
            .offset = 0,
            .size = @sizeOf(GraphicsArgs),
        };

        const pipeline_layout = try ctx.vkd.createPipelineLayout(ctx.dev, &vk.PipelineLayoutCreateInfo{
            .set_layout_count = 1,
            .p_set_layouts = &[_]vk.DescriptorSetLayout{layout},

            .push_constant_range_count = 1,
            .p_push_constant_ranges = &[_]vk.PushConstantRange{pushConstantRange},
        }, null);
        errdefer ctx.vkd.destroyPipelineLayout(ctx.dev, pipeline_layout, null);

        const render_pass = try createRenderPass(ctx, swapchain);

        const framebuffers = try createFramebuffers(ctx, allocator, render_pass, swapchain);

        return GraphicsPipe{
            .ctx = ctx,
            .allocator = allocator,

            .pipeline_layout = pipeline_layout,
            .descriptor_pool = pool,
            .descriptor_layout = layout,
            .descriptors = descriptors,

            .extent = swapchain.extent,
            .framebuffers = framebuffers,
            .render_pass = render_pass,
            .vertex = vertex,
            .fragment = fragment,
        };
    }

    pub fn resize(self: *GraphicsPipe, swapchain: *Swapchain) !void {
        self.deinitFramebuffers();
        self.extent = swapchain.extent;
        self.framebuffers = try createFramebuffers(self.ctx, self.allocator, self.render_pass, swapchain);
    }

    fn deinitFramebuffers(self: *GraphicsPipe) void {
        for (self.framebuffers) |fb| {
            self.ctx.vkd.destroyFramebuffer(self.ctx.dev, fb, null);
        }
        self.allocator.free(self.framebuffers);
    }

    pub fn deinit(self: *GraphicsPipe) void {
        self.ctx.vkd.destroyPipelineLayout(self.ctx.dev, self.pipeline_layout, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.vertex, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.fragment, null);

        self.ctx.vkd.destroyDescriptorPool(self.ctx.dev, self.descriptor_pool, null);
        self.ctx.vkd.destroyDescriptorSetLayout(self.ctx.dev, self.descriptor_layout, null);
        self.allocator.free(self.descriptors);

        self.deinitFramebuffers();
        self.ctx.vkd.destroyRenderPass(self.ctx.dev, self.render_pass, null);
    }

    pub fn begin(self: *const GraphicsPipe, cmdbuf: vk.CommandBuffer, frame: usize) void {
        self.ctx.vkd.cmdSetViewport(cmdbuf, 0, 1, &[_]vk.Viewport{
            .{
                .x = 0,
                .y = 0,
                .width = @as(f32, @floatFromInt(self.extent.width)),
                .height = @as(f32, @floatFromInt(self.extent.height)),
                .min_depth = 0,
                .max_depth = 1,
            },
        });

        // This needs to be a separate definition - see https://github.com/ziglang/zig/issues/7627.
        const render_area = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.extent,
        };

        const clear = vk.ClearValue{
            .color = .{ .float_32 = .{ 0.0, 0.0, 0.01, 1 } },
        };

        self.ctx.vkd.cmdBeginRenderPass(cmdbuf, &.{
            .render_pass = self.render_pass,
            .framebuffer = self.framebuffers[frame],
            .render_area = render_area,
            .clear_value_count = 1,
            .p_clear_values = @as([*]const vk.ClearValue, @ptrCast(&clear)),
        }, .@"inline");
    }

    pub fn end(self: *const GraphicsPipe, cmdbuf: vk.CommandBuffer) void {
        self.ctx.vkd.cmdEndRenderPass(cmdbuf);
    }
};

fn createFramebuffers(gc: *const VkContext, allocator: std.mem.Allocator, render_pass: vk.RenderPass, swapchain: *Swapchain) ![]vk.Framebuffer {
    const framebuffers = try allocator.alloc(vk.Framebuffer, swapchain.swap_images.len);
    errdefer allocator.free(framebuffers);

    var i: usize = 0;
    errdefer for (framebuffers[0..i]) |fb| gc.vkd.destroyFramebuffer(gc.dev, fb, null);

    for (framebuffers) |*fb| {
        fb.* = try gc.vkd.createFramebuffer(gc.dev, &vk.FramebufferCreateInfo{
            .flags = .{},
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&swapchain.swap_images[i].view),
            .width = swapchain.extent.width,
            .height = swapchain.extent.height,
            .layers = 1,
        }, null);
        i += 1;
    }

    return framebuffers;
}

fn createRenderPass(gc: *const VkContext, swapchain: *Swapchain) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .flags = .{},
        .format = swapchain.surface_format.format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };

    const color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };

    const subpass = vk.SubpassDescription{
        .flags = .{},
        .pipeline_bind_point = .graphics,
        .input_attachment_count = 0,
        .p_input_attachments = undefined,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment_ref),
        .p_resolve_attachments = null,
        .p_depth_stencil_attachment = null,
        .preserve_attachment_count = 0,
        .p_preserve_attachments = undefined,
    };

    return try gc.vkd.createRenderPass(gc.dev, &vk.RenderPassCreateInfo{
        .flags = .{},
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 0,
        .p_dependencies = undefined,
    }, null);
}

pub fn createPipeline(
    ctx: *const VkContext,
    layout: vk.PipelineLayout,
    render_pass: vk.RenderPass,
    vertex_shader: vk.ShaderModule,
    fragment_shader: vk.ShaderModule,
) !vk.Pipeline {
    const pssci = [_]vk.PipelineShaderStageCreateInfo{
        .{
            .flags = .{},
            .stage = .{ .vertex_bit = true },
            .module = vertex_shader,
            .p_name = "main",
            .p_specialization_info = null,
        },
        .{
            .flags = .{},
            .stage = .{ .fragment_bit = true },
            .module = fragment_shader,
            .p_name = "main",
            .p_specialization_info = null,
        },
    };

    const pvisci = vk.PipelineVertexInputStateCreateInfo{
        .flags = .{},
        .vertex_binding_description_count = 0,
        .p_vertex_binding_descriptions = null,
        .vertex_attribute_description_count = 0,
        .p_vertex_attribute_descriptions = null,
    };

    const piasci = vk.PipelineInputAssemblyStateCreateInfo{
        .flags = .{},
        .topology = .triangle_list,
        .primitive_restart_enable = vk.FALSE,
    };

    const pvsci = vk.PipelineViewportStateCreateInfo{
        .flags = .{},
        .viewport_count = 1,
        .p_viewports = undefined, // set in createCommandBuffers with cmdSetViewport
        .scissor_count = 1,
        .p_scissors = undefined, // set in createCommandBuffers with cmdSetScissor
    };

    const prsci = vk.PipelineRasterizationStateCreateInfo{
        .flags = .{},
        .depth_clamp_enable = vk.FALSE,
        .rasterizer_discard_enable = vk.FALSE,
        .polygon_mode = .fill,
        .cull_mode = .{ .back_bit = false },
        .front_face = .clockwise,
        .depth_bias_enable = vk.FALSE,
        .depth_bias_constant_factor = 0,
        .depth_bias_clamp = 0,
        .depth_bias_slope_factor = 0,
        .line_width = 1,
    };

    const pmsci = vk.PipelineMultisampleStateCreateInfo{
        .flags = .{},
        .rasterization_samples = .{ .@"1_bit" = true },
        .sample_shading_enable = vk.FALSE,
        .min_sample_shading = 1,
        .p_sample_mask = null,
        .alpha_to_coverage_enable = vk.FALSE,
        .alpha_to_one_enable = vk.FALSE,
    };

    const pcbas = vk.PipelineColorBlendAttachmentState{
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pcbsci = vk.PipelineColorBlendStateCreateInfo{
        .flags = .{},
        .logic_op_enable = vk.FALSE,
        .logic_op = .copy,
        .attachment_count = 1,
        .p_attachments = @as([*]const vk.PipelineColorBlendAttachmentState, @ptrCast(&pcbas)),
        .blend_constants = [_]f32{ 0, 0, 0, 0 },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };
    const pdsci = vk.PipelineDynamicStateCreateInfo{
        .flags = .{},
        .dynamic_state_count = dynstate.len,
        .p_dynamic_states = &dynstate,
    };

    const gpci = vk.GraphicsPipelineCreateInfo{
        .flags = .{},
        .stage_count = 2,
        .p_stages = &pssci,
        .p_vertex_input_state = &pvisci,
        .p_input_assembly_state = &piasci,
        .p_tessellation_state = null,
        .p_viewport_state = &pvsci,
        .p_rasterization_state = &prsci,
        .p_multisample_state = &pmsci,
        .p_depth_stencil_state = null,
        .p_color_blend_state = &pcbsci,
        .p_dynamic_state = &pdsci,
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    var pipeline: vk.Pipeline = undefined;
    _ = try ctx.vkd.createGraphicsPipelines(
        ctx.dev,
        .null_handle,
        1,
        @as([*]const vk.GraphicsPipelineCreateInfo, @ptrCast(&gpci)),
        null,
        @as([*]vk.Pipeline, @ptrCast(&pipeline)),
    );
    return pipeline;
}

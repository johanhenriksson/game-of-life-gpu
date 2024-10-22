const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");

const Color = @import("primitives.zig").Color;
const Rect = @import("primitives.zig").Rect;
const Vec2 = @import("primitives.zig").Vec2;
const Pattern = @import("pattern.zig").Pattern;

const VkContext = @import("context.zig").VkContext;
const Shader = @import("shader.zig");
const GraphicsPipe = @import("graphics_pipe.zig").GraphicsPipe;
const GraphicsArgs = @import("graphics_pipe.zig").GraphicsArgs;
const Cell = @import("pattern.zig").Cell;
const createPipeline = @import("graphics_pipe.zig").createPipeline;

const transitionImage = @import("helper.zig").transitionImage;
const blitImage = @import("helper.zig").blitImage;

pub const Cursor = struct {
    width: usize,
    height: usize,
    position: Vec2,

    image: vk.Image,
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    layout: vk.SubresourceLayout,
    size: usize,
    ptr: ?*anyopaque,

    ctx: *const VkContext,

    pub fn init(ctx: *const VkContext, pool: vk.CommandPool, size: usize) !Cursor {
        // create an image and allocate gpu memory
        const image = try ctx.vkd.createImage(ctx.dev, &.{
            .flags = .{},
            .image_type = .@"2d",
            .format = .r8g8b8a8_unorm,
            .extent = .{
                .width = @intCast(size),
                .height = @intCast(size),
                .depth = 1,
            },
            .mip_levels = 1,
            .array_layers = 1,
            .samples = .{ .@"1_bit" = true },
            .tiling = .linear,
            .usage = .{ .transfer_src_bit = true, .sampled_bit = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
            .initial_layout = .undefined,
        }, null);
        const mem_req = ctx.vkd.getImageMemoryRequirements(ctx.dev, image);
        const memory = try ctx.allocate(mem_req, .{ .host_visible_bit = true, .host_coherent_bit = true });
        try ctx.vkd.bindImageMemory(ctx.dev, image, memory, 0);

        // grab the image layout
        const subresource = vk.ImageSubresource{
            .aspect_mask = .{ .color_bit = true },
            .mip_level = 0,
            .array_layer = 0,
        };
        const layout = ctx.vkd.getImageSubresourceLayout(ctx.dev, image, &subresource);

        // create a buffer object over the image memory
        const buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
            .flags = .{},
            .size = mem_req.size,
            .usage = .{ .transfer_src_bit = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        }, null);
        try ctx.vkd.bindBufferMemory(ctx.dev, buffer, memory, 0);

        // get a pointer to the gpu memory
        const ptr = try ctx.vkd.mapMemory(ctx.dev, memory, 0, vk.WHOLE_SIZE, .{});

        try transitionImage(ctx, pool, image, .undefined, .general);

        return .{
            .ctx = ctx,
            .width = 0,
            .height = 0,
            .size = size,
            .layout = layout,
            .image = image,
            .buffer = buffer,
            .memory = memory,
            .ptr = ptr,
            .position = Vec2{ .x = 0, .y = 0 },
        };
    }

    pub fn deinit(self: *const Cursor) void {
        self.ctx.vkd.destroyImage(self.ctx.dev, self.image, null);
        self.ctx.vkd.unmapMemory(self.ctx.dev, self.memory);
        self.ctx.vkd.freeMemory(self.ctx.dev, self.memory, null);
        self.ctx.vkd.destroyBuffer(self.ctx.dev, self.buffer, null);
    }

    pub fn clear(self: *Cursor) !void {
        for (0..self.height) |y| {
            for (0..self.width) |x| {
                try self.set(x, y, .dead);
            }
        }
    }

    pub fn resize(self: *Cursor, width: usize, height: usize) !void {
        if (width < 0 or width > self.size or height < 0 or height > self.size) {
            return error.OutOfBounds;
        }
        self.width = width;
        self.height = height;
        try self.clear();
    }

    pub fn set(self: *Cursor, x: usize, y: usize, cell: Cell) !void {
        if (x < 0 or y < 0 or x >= self.size or y >= self.size) {
            return error.OutOfBounds;
        }
        const color: Color = if (cell == .alive) Color.rgb(255, 255, 0) else Color.rgb(0, 0, 0);
        const row_start = self.layout.offset + (y * self.layout.row_pitch);
        const pixel_offset = x * @sizeOf(Color);
        const bytes: [*]u8 = @ptrCast(self.ptr);
        const color_bytes: *const [@sizeOf(Color)]u8 = @ptrCast(&color);
        @memcpy(bytes[row_start + pixel_offset ..][0..@sizeOf(Color)], color_bytes);
    }

    pub fn setPattern(self: *Cursor, pattern: *const Pattern) !void {
        std.debug.print("set cursor pattern: {s}\n", .{pattern.name});
        try self.resize(pattern.width, pattern.height);
        for (0..pattern.height) |y| {
            for (0..pattern.width) |x| {
                const cell = try pattern.get(x, y);
                try self.set(x, y, cell);
            }
        }
    }

    pub fn setPosition(self: *Cursor, position: Vec2) void {
        self.position = position;
    }

    pub fn paste(self: *const Cursor, pool: vk.CommandPool, dst_image: vk.Image, x: i32, y: i32) !void {
        // copy cursor to compute initial buffer (which is the last frame)
        const src_rect = Rect.init(0, 0, @intCast(self.width), @intCast(self.height));
        const dst_rect = Rect.init(x, y, @intCast(self.width), @intCast(self.height));

        try blitImage(self.ctx, pool, self.image, dst_image, src_rect, dst_rect);
    }
};

pub const CursorView = struct {
    ctx: *const VkContext,
    gfx: *const GraphicsPipe,

    cursor: *const Cursor,

    view: vk.ImageView,
    sampler: vk.Sampler,
    pipeline: vk.Pipeline,
    shader: vk.ShaderModule,
    descriptors: []vk.DescriptorSet,

    pub fn init(ctx: *const VkContext, allocator: std.mem.Allocator, cursor: *const Cursor, gfx: *GraphicsPipe) !CursorView {
        const view = try ctx.vkd.createImageView(ctx.dev, &.{
            .image = cursor.image,
            .view_type = .@"2d",
            .format = .r8g8b8a8_unorm,
            .components = .{
                .r = .r,
                .g = .r,
                .b = .r,
                .a = .r,
            },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);

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
        const descriptors = try allocator.alloc(vk.DescriptorSet, 3);
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
                            .image_view = view,
                            .image_layout = .shader_read_only_optimal,
                        },
                    },
                    .p_buffer_info = undefined,
                    .p_texel_buffer_view = undefined,
                },
            }, 0, null);
        }

        const shader = try Shader.compile(ctx, allocator, Shader.Stage.fragment, "shaders/cursor.fs.glsl");

        const pipeline = try createPipeline(ctx, gfx.pipeline_layout, gfx.render_pass, gfx.vertex, shader);

        return CursorView{
            .ctx = ctx,
            .gfx = gfx,
            .cursor = cursor,

            .view = view,
            .sampler = sampler,
            .shader = shader,
            .pipeline = pipeline,
            .descriptors = descriptors,
        };
    }

    pub fn deinit(self: *const CursorView) void {
        self.ctx.vkd.destroySampler(self.ctx.dev, self.sampler, null);
        self.ctx.vkd.destroyImageView(self.ctx.dev, self.view, null);
        // todo: free descriptor sets somehow
        self.ctx.vkd.destroyPipeline(self.ctx.dev, self.pipeline, null);
        self.ctx.vkd.destroyShaderModule(self.ctx.dev, self.shader, null);
    }

    pub fn draw(self: *CursorView, cmdbuf: vk.CommandBuffer, view: zm.Mat, frame: usize) void {
        const pos = self.cursor.position;
        const model = zm.translation(@floor(pos.x), @floor(pos.y), 0);

        self.ctx.vkd.cmdBindPipeline(cmdbuf, .graphics, self.pipeline);

        self.ctx.vkd.cmdPushConstants(cmdbuf, self.gfx.pipeline_layout, .{ .vertex_bit = true, .fragment_bit = true }, 0, @sizeOf(GraphicsArgs), @ptrCast(&GraphicsArgs{
            .proj = view,
            .model = model,
            .size = Vec2{
                .x = @floatFromInt(self.cursor.width),
                .y = @floatFromInt(self.cursor.height),
            },
            .tex_scale = Vec2{
                .x = @as(f32, @floatFromInt(self.cursor.width)) / @as(f32, @floatFromInt(self.cursor.size)),
                .y = @as(f32, @floatFromInt(self.cursor.height)) / @as(f32, @floatFromInt(self.cursor.size)),
            },
        }));
        self.ctx.vkd.cmdBindDescriptorSets(cmdbuf, .graphics, self.gfx.pipeline_layout, 0, 1, @as([*]const vk.DescriptorSet, @ptrCast(&self.descriptors[frame])), 0, null);
        self.ctx.vkd.cmdDraw(cmdbuf, 6, 1, 0, 0);
    }
};

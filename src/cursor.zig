const Color = @import("primitives.zig").Color;
const Rect = @import("primitives.zig").Rect;
const VkContext = @import("context.zig").VkContext;
const vk = @import("vulkan");

const transitionImage = @import("helper.zig").transitionImage;
const blitImage = @import("helper.zig").blitImage;

pub const Cursor = struct {
    pixels: []Color,
    width: i32,
    height: i32,

    image: vk.Image,
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,

    ctx: *const VkContext,

    pub fn init(ctx: *const VkContext, pool: vk.CommandPool, width: i32, height: i32) !Cursor {
        const image = try ctx.vkd.createImage(ctx.dev, &.{
            .flags = .{},
            .image_type = vk.ImageType.@"2d",
            .format = vk.Format.r8g8b8a8_unorm,
            .extent = vk.Extent3D{ .width = @intCast(width), .height = @intCast(height), .depth = 1 },
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
        const mem_req = ctx.vkd.getImageMemoryRequirements(ctx.dev, image);
        const memory = try ctx.allocate(mem_req, .{ .host_visible_bit = true, .host_coherent_bit = true });
        try ctx.vkd.bindImageMemory(ctx.dev, image, memory, 0);

        const buffer = try ctx.vkd.createBuffer(ctx.dev, &.{
            .flags = .{},
            .size = mem_req.size,
            .usage = .{ .transfer_src_bit = true },
            .sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = undefined,
        }, null);
        try ctx.vkd.bindBufferMemory(ctx.dev, buffer, memory, 0);

        const cursor_image_ptr = try ctx.vkd.mapMemory(ctx.dev, memory, 0, vk.WHOLE_SIZE, .{});
        const cursor_image_colors: [*]Color = @ptrCast(@alignCast(cursor_image_ptr));
        const pixels: []Color = cursor_image_colors[0..@intCast(width * height)];

        try transitionImage(ctx, pool, image, vk.ImageLayout.undefined, vk.ImageLayout.general);

        return Cursor{
            .ctx = ctx,
            .pixels = pixels,
            .width = width,
            .height = height,
            .image = image,
            .buffer = buffer,
            .memory = memory,
        };
    }

    pub fn deinit(self: *const Cursor) void {
        self.ctx.vkd.unmapMemory(self.ctx.dev, self.memory);
        self.ctx.vkd.destroyBuffer(self.ctx.dev, self.buffer, null);
        self.ctx.vkd.freeMemory(self.ctx.dev, self.memory, null);
    }

    pub fn set(self: *const Cursor, x: i32, y: i32, color: Color) void {
        self.pixels[@intCast(y * self.width + x)] = color;
    }

    pub fn paste(self: *const Cursor, pool: vk.CommandPool, dst_image: vk.Image, x: i32, y: i32) !void {
        // copy cursor to compute initial buffer (which is the last frame)
        const src_rect = Rect.init(0, 0, self.width, self.height);
        const dst_rect = Rect.init(x, y, self.width, self.height);

        try blitImage(self.ctx, pool, self.image, dst_image, src_rect, dst_rect);
    }
};

const Color = @import("primitives.zig").Color;
const Rect = @import("primitives.zig").Rect;
const Pattern = @import("pattern.zig").Pattern;

const VkContext = @import("context.zig").VkContext;
const vk = @import("vulkan");

const transitionImage = @import("helper.zig").transitionImage;
const blitImage = @import("helper.zig").blitImage;

pub const Cursor = struct {
    pixels: []Color,
    width: usize,
    height: usize,

    image: vk.Image,
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    max_width: usize,
    max_height: usize,

    ctx: *const VkContext,

    pub fn init(ctx: *const VkContext, pool: vk.CommandPool, max_width: usize, max_height: usize) !Cursor {
        const image = try ctx.vkd.createImage(ctx.dev, &.{
            .flags = .{},
            .image_type = vk.ImageType.@"2d",
            .format = vk.Format.r8g8b8a8_unorm,
            .extent = vk.Extent3D{ .width = @intCast(max_width), .height = @intCast(max_height), .depth = 1 },
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
        const pixels: []Color = cursor_image_colors[0..@intCast(max_width * max_height)];

        try transitionImage(ctx, pool, image, vk.ImageLayout.undefined, vk.ImageLayout.general);

        return Cursor{
            .ctx = ctx,
            .pixels = pixels,
            .width = 0,
            .height = 0,
            .max_width = max_width,
            .max_height = max_height,
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

    pub fn clear(self: *Cursor) void {
        for (0..self.pixels.len) |i| {
            self.pixels[i] = Color.rgb(0, 0, 0);
        }
        self.width = 0;
        self.height = 0;
    }

    pub fn set(self: *Cursor, x: usize, y: usize, color: Color) !void {
        if (x < 0 or x >= self.max_width or y < 0 or y >= self.max_height) {
            return error.OutOfBounds;
        }
        self.width = @max(self.width, x + 1);
        self.height = @max(self.height, y + 1);
        self.pixels[y * self.max_width + x] = color;
    }

    pub fn setPattern(self: *Cursor, pattern: *const Pattern) !void {
        if (pattern.width > self.max_width or pattern.height > self.max_height) {
            return error.OutOfBounds;
        }
        self.clear();
        for (0..pattern.height) |y| {
            for (0..pattern.width) |x| {
                const cell = try pattern.get(x, y);
                if (cell == .alive) {
                    try self.set(x, y, Color.rgb(255, 255, 255));
                } else {
                    try self.set(x, y, Color.rgb(0, 0, 0));
                }
            }
        }
    }

    pub fn paste(self: *const Cursor, pool: vk.CommandPool, dst_image: vk.Image, x: i32, y: i32) !void {
        // copy cursor to compute initial buffer (which is the last frame)
        const src_rect = Rect.init(0, 0, @intCast(self.width), @intCast(self.height));
        const dst_rect = Rect.init(x, y, @intCast(self.width), @intCast(self.height));

        try blitImage(self.ctx, pool, self.image, dst_image, src_rect, dst_rect);
    }
};

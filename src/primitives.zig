const vk = @import("vulkan");

pub const Vec2 = struct {
    x: f32,
    y: f32,
};

pub const Color = struct {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
};

pub const Rect = struct {
    x: i32,
    y: i32,
    w: i32,
    h: i32,

    pub fn init(x: i32, y: i32, w: i32, h: i32) Rect {
        return Rect{ .x = x, .y = y, .w = w, .h = h };
    }

    pub fn offsets3D(self: *const Rect) [2]vk.Offset3D {
        return [2]vk.Offset3D{
            .{ .x = self.x, .y = self.y, .z = 0 },
            .{ .x = self.x + self.w, .y = self.y + self.h, .z = 1 },
        };
    }
};

const std = @import("std");
const vk = @import("vulkan");
const zm = @import("zmath");

const VkContext = @import("context.zig").VkContext;
const Vec2 = @import("primitives.zig").Vec2;

pub const Viewport = struct {
    ctx: *const VkContext,
    width: f32,
    height: f32,

    position: zm.F32x4,
    scale: f32,

    pub fn init(ctx: *const VkContext, width: f32, height: f32, x: f32, y: f32) !Viewport {
        return Viewport{
            .ctx = ctx,
            .width = width,
            .height = height,
            .position = zm.f32x4(x, y, 0, 0),
            .scale = 5,
        };
    }

    pub fn deinit(self: *const Viewport) void {
        _ = self;
    }

    pub fn resize(self: *Viewport, width: f32, height: f32) void {
        self.width = width;
        self.height = height;
    }

    pub fn zoom(self: *Viewport, delta: f32) void {
        const sign: f32 = if (delta >= 0) 1.1 else 0.9;
        self.scale *= sign;
        self.scale = zm.clamp(self.scale, 0.1, 20);
    }

    pub fn pan(self: *Viewport, x: f32, y: f32) void {
        self.position[0] += x / self.scale;
        self.position[1] += y / self.scale;
    }

    fn projMatrix(self: *const Viewport) zm.Mat {
        const w = self.width / self.scale;
        const h = self.height / self.scale;
        return zm.orthographicLh(w, h, 0, 2);
    }

    fn viewMatrix(self: *const Viewport) zm.Mat {
        return zm.translation(-self.position[0], -self.position[1], 0);
    }

    pub fn matrix(self: *const Viewport) zm.Mat {
        return zm.mul(self.viewMatrix(), self.projMatrix());
    }

    pub fn screenToWorld(self: *const Viewport, screen: Vec2) Vec2 {
        // screen coordinates -> ndc
        const ndc = zm.f32x4(2 * (screen.x) / self.width - 1, 2 * (screen.y) / self.height - 1, 0, 1);

        // ndc -> world
        const world = zm.mul(ndc, zm.inverse(self.matrix()));

        return Vec2{ .x = world[0], .y = world[1] };
    }
};

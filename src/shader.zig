const std = @import("std");
const vk = @import("vulkan");
const VkContext = @import("context.zig").VkContext;

pub const Stage = enum {
    vertex,
    fragment,
    compute,
};

pub fn compile(ctx: *const VkContext, allocator: std.mem.Allocator, stage: Stage, input_file: []const u8) !vk.ShaderModule {
    const stageFlag = switch (stage) {
        Stage.vertex => "-fshader-stage=vertex",
        Stage.fragment => "-fshader-stage=fragment",
        Stage.compute => "-fshader-stage=compute",
    };

    const args = [_][]const u8{
        "glslc",
        "--target-env=vulkan1.1",
        stageFlag,
        "-o",
        "-", // output to stdout
        input_file,
    };

    const result = try std.process.Child.run(.{
        .allocator = allocator,
        .argv = &args,
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .Exited => |code| {
            if (code != 0) {
                std.debug.print("glslc exited with non-zero status code: {d}\n", .{code});
                std.debug.print("stderr: {s}\n", .{result.stderr});
                return error.GlslcFailed;
            }
        },
        else => {
            std.debug.print("glslc failed to run\n", .{});
            return error.GlslcFailed;
        },
    }

    std.debug.print("compiled shader {s}.\n", .{input_file});

    return try ctx.vkd.createShaderModule(ctx.dev, &.{
        .code_size = result.stdout.len,
        .p_code = @alignCast(@ptrCast(result.stdout.ptr)),
    }, null);
}

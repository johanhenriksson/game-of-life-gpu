const std = @import("std");
const sdl = @import("sdl");

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) !void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "sdlvk",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // generate Vulkan bindings.
    const registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml");
    const vkzig_dep = b.dependency("vulkan_zig", .{
        .registry = registry.getPath(b),
    });
    const vkzig_bindings = vkzig_dep.module("vulkan-zig");
    exe.root_module.addImport("vulkan", vkzig_bindings);

    // sdl
    const sdl_dep = sdl.init(b, "sdl");
    sdl_dep.link(exe, .dynamic);
    exe.root_module.addImport("sdl2", sdl_dep.getWrapperModuleVulkan(vkzig_bindings));

    // Compile shaders at build time so that they can be imported with '@embedFile'.
    try compileShader(b, exe, "triangle.vert");
    try compileShader(b, exe, "triangle.frag");

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}

fn compileShader(b: *std.Build, exe: *std.Build.Step.Compile, name: []const u8) !void {
    const inputPath = try std.fmt.allocPrint(b.allocator, "shaders/{s}", .{name});
    const outputPath = try std.fmt.allocPrint(b.allocator, "{s}.spv", .{name});

    const glslc = b.addSystemCommand(&.{"glslc"});
    glslc.addFileArg(b.path(inputPath));
    glslc.addArgs(&.{ "--target-env=vulkan1.1", "-o" });
    const spv_file = glslc.addOutputFileArg(outputPath);

    exe.root_module.addAnonymousImport(name, .{
        .root_source_file = spv_file,
    });
}

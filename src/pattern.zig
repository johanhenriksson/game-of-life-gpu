const std = @import("std");
const Color = @import("primitives.zig").Color;

pub const Cell = enum(u8) {
    dead = 0,
    alive = 1,
};

pub const Pattern = struct {
    name: []const u8,

    cells: []Cell,
    width: usize,
    height: usize,

    pub fn loadDir(allocator: std.mem.Allocator, dir_path: []const u8) ![]Pattern {
        var dir = try std.fs.cwd().openDir(dir_path, .{ .iterate = true });
        defer dir.close();

        var file_list = std.ArrayList([]const u8).init(allocator);
        defer {
            for (file_list.items) |item| {
                allocator.free(item);
            }
            file_list.deinit();
        }

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind == .file) {
                const file_name = try allocator.dupe(u8, entry.name);
                try file_list.append(file_name);
            }
        }

        var pattern_list = std.ArrayList(Pattern).init(allocator);
        errdefer pattern_list.deinit();

        for (file_list.items) |file_name| {
            const file_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, file_name });
            defer allocator.free(file_path);
            const pattern = try Pattern.loadFile(allocator, file_path);
            std.debug.print("{s}\n", .{file_path});
            try pattern_list.append(pattern);
        }

        return pattern_list.toOwnedSlice();
    }

    pub fn loadFile(allocator: std.mem.Allocator, file_path: []const u8) !Pattern {
        const file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        // Read the entire file content
        const file_content = try file.readToEndAlloc(allocator, 1024 * 1024); // 1MB limit
        defer allocator.free(file_content);

        return try Pattern.load(allocator, file_path, file_content);
    }

    pub fn load(allocator: std.mem.Allocator, name: []const u8, input: []const u8) !Pattern {
        var cells = std.ArrayList(Cell).init(allocator);
        defer cells.deinit();

        var lines = std.ArrayList([]const u8).init(allocator);
        defer lines.deinit();

        const trimmed = std.mem.trim(u8, input, "\r\n");

        var width: usize = 0;

        // First pass: collect lines and find max length
        var line_iter = std.mem.split(u8, trimmed, "\n");
        while (line_iter.next()) |line| {
            if (std.mem.startsWith(u8, line, "!")) {
                continue; // Skip metadata lines
            }
            const tokens = std.mem.trim(u8, line, "\r\n");
            try lines.append(tokens);
            width = @max(width, tokens.len);
        }

        const height = lines.items.len;

        // Second pass: process lines and pad
        for (lines.items) |line| {
            for (line) |char| {
                switch (char) {
                    'O' => try cells.append(.alive),
                    '*' => try cells.append(.alive),
                    '.' => try cells.append(.dead),
                    else => return error.Unknown,
                }
            }
            // Pad with dead cells
            for (line.len..width) |_| {
                try cells.append(.dead);
            }
        }

        return Pattern{
            .name = try allocator.dupe(u8, name),
            .cells = try cells.toOwnedSlice(),
            .width = width,
            .height = height,
        };
    }

    pub fn get(self: *const Pattern, x: usize, y: usize) !Cell {
        if (x >= self.width or y >= self.height) {
            return error.OutOfBounds;
        }
        return self.cells[self.width * y + x];
    }

    pub fn deinit(self: *Pattern, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.cells);
    }

    pub fn print(self: *const Pattern) void {
        for (0..self.height) |y| {
            for (0..self.width) |x| {
                const cell = self.cells[self.width * y + x];
                switch (cell) {
                    .dead => std.debug.print(".", .{}),
                    .alive => std.debug.print("O", .{}),
                }
            }
            std.debug.print("\n", .{});
        }
    }
};

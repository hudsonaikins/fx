const std = @import("std");
const stream_provider = @import("../core/agent/stream_provider.zig");
const model_tool_schema = @import("../core/tooling/model_tool_schema.zig");
const io_mod = @import("../core/shared/io.zig");
const types = @import("../core/shared/types.zig");
const gateway_client = @import("client.zig");

const Allocator = std.mem.Allocator;

pub const default_model = "lfm2.5-8b-a1b@q4_k_m";
const default_chat_url = "http://127.0.0.1:1234/v1/chat/completions";
const chat_url_env = "FX_LOCAL_CHAT_URL";
const allow_non_loopback_env = "FX_LOCAL_ALLOW_NON_LOOPBACK";
const api_key_env = "FX_LOCAL_API_KEY";
const max_error_body_bytes: usize = 256 * 1024;
const max_sse_line_bytes: usize = 32 * 1024 * 1024;
const max_sse_aggregate_bytes: usize = 64 * 1024 * 1024;
const max_sse_events: usize = 100_000;
const max_tool_calls: usize = 128;
const max_tool_identity_bytes: usize = 1024;
const max_tool_arguments_bytes: usize = 4 * 1024 * 1024;
const transfer_buffer_bytes: usize = 256 * 1024;
const connect_timeout_ms: i64 = 30_000;

pub const agent_stream_provider = stream_provider.Provider{
    .stream_fn = streamCompletion,
};

fn validateModel(model: []const u8) !void {
    if (model.len == 0 or model.len > 1024) return error.InvalidLocalModel;
    for (model) |byte| {
        if (byte <= 0x20 or byte == 0x7f) return error.InvalidLocalModel;
    }
}

fn validateImages(request: stream_provider.RequestData) !void {
    if (request.verified_images != null) return error.LocalProviderVisionUnsupported;
    for (request.messages) |message| {
        if (message.images.len > 0) return error.LocalProviderVisionUnsupported;
    }
}

fn containsName(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}

fn writeFunctionTool(
    writer: *std.Io.Writer,
    alloc: Allocator,
    name: []const u8,
    description: []const u8,
    schema: model_tool_schema.ObjectSchema,
) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
    try std.json.Stringify.value(name, .{}, writer);
    try writer.writeAll(",\"description\":");
    try model_tool_schema.writeCappedDescriptionJsonString(alloc, writer, description);
    try writer.writeAll(",\"parameters\":");
    try model_tool_schema.writeObjectSchema(alloc, writer, schema);
    try writer.writeAll("}}");
}

fn writeDynamicFunctionTool(
    writer: *std.Io.Writer,
    name: []const u8,
    description: []const u8,
    schema: std.json.Value,
) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":");
    try std.json.Stringify.value(name, .{}, writer);
    try writer.writeAll(",\"description\":");
    try std.json.Stringify.value(description, .{}, writer);
    try writer.writeAll(",\"parameters\":");
    try std.json.Stringify.value(schema, .{}, writer);
    try writer.writeAll("}}");
}

fn writeTools(writer: *std.Io.Writer, alloc: Allocator, tools: stream_provider.ToolSelection) !usize {
    var count: usize = 0;
    try writer.writeAll("[");
    for (tools.advertised_names) |name| {
        const tool = tools.advertisedFunction(name) orelse continue;
        if (count > 0) try writer.writeByte(',');
        try writeFunctionTool(writer, alloc, tool.name, tool.description, tool.input_schema);
        count += 1;
    }
    for (tools.additional_functions) |tool| {
        if (containsName(tools.advertised_names, tool.name)) continue;
        if (count > 0) try writer.writeByte(',');
        try writeFunctionTool(writer, alloc, tool.name, tool.description, tool.input_schema);
        count += 1;
    }
    for (tools.selected_dynamic) |tool| {
        if (containsName(tools.advertised_names, tool.name)) continue;
        if (count > 0) try writer.writeByte(',');
        try writeDynamicFunctionTool(writer, tool.name, tool.description, tool.input_schema);
        count += 1;
    }
    try writer.writeByte(']');
    return count;
}

fn writeMessage(writer: *std.Io.Writer, message: types.ChatMessage) !void {
    try writer.writeAll("{\"role\":");
    try std.json.Stringify.value(@tagName(message.role), .{}, writer);
    switch (message.role) {
        .system, .user => {
            try writer.writeAll(",\"content\":");
            try std.json.Stringify.value(message.content orelse "", .{}, writer);
        },
        .assistant => {
            if (message.tool_calls.len == 0) {
                try writer.writeAll(",\"content\":");
                try std.json.Stringify.value(message.content orelse "", .{}, writer);
            } else {
                try writer.writeAll(",\"content\":null,\"tool_calls\":[");
                for (message.tool_calls, 0..) |call, index| {
                    if (index > 0) try writer.writeByte(',');
                    try writer.writeAll("{\"id\":");
                    try std.json.Stringify.value(call.id, .{}, writer);
                    try writer.writeAll(",\"type\":\"function\",\"function\":{\"name\":");
                    try std.json.Stringify.value(call.name, .{}, writer);
                    try writer.writeAll(",\"arguments\":");
                    try std.json.Stringify.value(call.arguments_json, .{}, writer);
                    try writer.writeAll("}}");
                }
                try writer.writeByte(']');
            }
        },
        .tool => {
            try writer.writeAll(",\"tool_call_id\":");
            try std.json.Stringify.value(message.tool_call_id orelse "", .{}, writer);
            try writer.writeAll(",\"content\":");
            try std.json.Stringify.value(message.content orelse "", .{}, writer);
        },
    }
    try writer.writeByte('}');
}

pub fn buildRequest(alloc: Allocator, request: stream_provider.RequestData) ![]u8 {
    try validateModel(request.model);
    try validateImages(request);
    if (request.budget) |budget| {
        if (budget.cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
    }

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.writeAll("{\"model\":");
    try std.json.Stringify.value(request.model, .{}, &out.writer);
    try out.writer.writeAll(",\"messages\":[");
    for (request.messages, 0..) |message, index| {
        if (message.images.len > 0) return error.LocalProviderVisionUnsupported;
        if (index > 0) try out.writer.writeByte(',');
        try writeMessage(&out.writer, message);
    }
    try out.writer.writeAll("],\"stream\":true,\"tools\":");
    const tool_count = try writeTools(&out.writer, alloc, request.tools);
    try out.writer.writeAll(",\"tool_choice\":");
    try std.json.Stringify.value(request.tool_choice.label(), .{}, &out.writer);
    if (tool_count > 0) try out.writer.writeAll(",\"parallel_tool_calls\":true");
    if (request.max_output_tokens) |limit| try out.writer.print(",\"max_tokens\":{d}", .{limit});
    if (request.response_format) |format| {
        if (format.schema != .object) return error.InvalidStructuredResponseSchema;
        try out.writer.writeAll(",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":");
        try std.json.Stringify.value(format.name, .{}, &out.writer);
        try out.writer.writeAll(",\"description\":");
        try std.json.Stringify.value(format.description, .{}, &out.writer);
        try out.writer.writeAll(",\"schema\":");
        try std.json.Stringify.value(format.schema, .{}, &out.writer);
        try out.writer.writeAll(",\"strict\":true}}");
    }
    try out.writer.writeByte('}');
    if (request.budget) |budget| {
        if (budget.cancel_flag) |flag| if (flag.load(.seq_cst)) return error.Cancelled;
    }
    return try out.toOwnedSlice();
}

const ToolAccumulator = struct {
    id: []u8 = &.{},
    name: []u8 = &.{},
    arguments: std.ArrayList(u8) = .empty,
    started: bool = false,

    fn deinit(self: *ToolAccumulator, alloc: Allocator) void {
        if (self.id.len > 0) alloc.free(self.id);
        if (self.name.len > 0) alloc.free(self.name);
        self.arguments.deinit(alloc);
        self.* = undefined;
    }
};

const fallback_tool_tags = [_]struct {
    open: []const u8,
    close: []const u8,
}{
    .{ .open = "<|tool_call_start|>", .close = "<|tool_call_end|>" },
    .{ .open = "<tool_call>", .close = "</tool_call>" },
};

const TaggedToolCall = struct {
    index: usize,
    open_len: usize,
    close: []const u8,
};

fn nextTaggedToolCall(content: []const u8, offset: usize) ?TaggedToolCall {
    var found: ?TaggedToolCall = null;
    for (fallback_tool_tags) |tag| {
        const relative = std.mem.indexOf(u8, content[offset..], tag.open) orelse continue;
        const candidate = TaggedToolCall{
            .index = offset + relative,
            .open_len = tag.open.len,
            .close = tag.close,
        };
        if (found == null or candidate.index < found.?.index) found = candidate;
    }
    return found;
}

fn isIdentifierStart(byte: u8) bool {
    return std.ascii.isAlphabetic(byte) or byte == '_';
}

fn isIdentifierContinue(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

const PythonValueParser = struct {
    input: []const u8,
    index: usize = 0,
    alloc: Allocator,
    writer: *std.Io.Writer,

    fn skipWhitespace(self: *PythonValueParser) void {
        while (self.index < self.input.len and std.ascii.isWhitespace(self.input[self.index])) : (self.index += 1) {}
    }

    fn expect(self: *PythonValueParser, byte: u8) !void {
        self.skipWhitespace();
        if (self.index >= self.input.len or self.input[self.index] != byte) {
            return error.InvalidLocalProviderNativeToolCall;
        }
        self.index += 1;
    }

    fn parseIdentifier(self: *PythonValueParser) ![]const u8 {
        self.skipWhitespace();
        if (self.index >= self.input.len or !isIdentifierStart(self.input[self.index])) {
            return error.InvalidLocalProviderNativeToolCall;
        }
        const start = self.index;
        self.index += 1;
        while (self.index < self.input.len and isIdentifierContinue(self.input[self.index])) : (self.index += 1) {}
        return self.input[start..self.index];
    }

    fn appendCodepoint(decoded: *std.ArrayList(u8), alloc: Allocator, codepoint: u21) !void {
        if (codepoint >= 0xd800 and codepoint <= 0xdfff) {
            return error.InvalidLocalProviderNativeToolCall;
        }
        if (codepoint <= 0x7f) {
            try decoded.append(alloc, @intCast(codepoint));
        } else if (codepoint <= 0x7ff) {
            try decoded.append(alloc, @intCast(0xc0 | (codepoint >> 6)));
            try decoded.append(alloc, @intCast(0x80 | (codepoint & 0x3f)));
        } else {
            try decoded.append(alloc, @intCast(0xe0 | (codepoint >> 12)));
            try decoded.append(alloc, @intCast(0x80 | ((codepoint >> 6) & 0x3f)));
            try decoded.append(alloc, @intCast(0x80 | (codepoint & 0x3f)));
        }
    }

    fn parseString(self: *PythonValueParser) !void {
        self.skipWhitespace();
        if (self.index >= self.input.len or (self.input[self.index] != '\'' and self.input[self.index] != '"')) {
            return error.InvalidLocalProviderNativeToolCall;
        }
        const quote = self.input[self.index];
        self.index += 1;

        var decoded: std.ArrayList(u8) = .empty;
        defer decoded.deinit(self.alloc);
        while (self.index < self.input.len) {
            const byte = self.input[self.index];
            self.index += 1;
            if (byte == quote) {
                try std.json.Stringify.value(decoded.items, .{}, self.writer);
                return;
            }
            if (byte != '\\') {
                try decoded.append(self.alloc, byte);
                continue;
            }
            if (self.index >= self.input.len) return error.InvalidLocalProviderNativeToolCall;
            const escaped = self.input[self.index];
            self.index += 1;
            switch (escaped) {
                '\\' => try decoded.append(self.alloc, '\\'),
                '\'' => try decoded.append(self.alloc, '\''),
                '"' => try decoded.append(self.alloc, '"'),
                'n' => try decoded.append(self.alloc, '\n'),
                'r' => try decoded.append(self.alloc, '\r'),
                't' => try decoded.append(self.alloc, '\t'),
                'b' => try decoded.append(self.alloc, '\x08'),
                'f' => try decoded.append(self.alloc, '\x0c'),
                'u' => {
                    if (self.input.len - self.index < 4) return error.InvalidLocalProviderNativeToolCall;
                    var codepoint: u21 = 0;
                    for (self.input[self.index .. self.index + 4]) |hex| {
                        const digit = std.fmt.charToDigit(hex, 16) catch return error.InvalidLocalProviderNativeToolCall;
                        codepoint = @as(u21, codepoint << 4) | @as(u21, digit);
                    }
                    self.index += 4;
                    try appendCodepoint(&decoded, self.alloc, codepoint);
                },
                else => return error.InvalidLocalProviderNativeToolCall,
            }
        }
        return error.InvalidLocalProviderNativeToolCall;
    }

    fn parseBare(self: *PythonValueParser) !void {
        const start = self.index;
        while (self.index < self.input.len) {
            const byte = self.input[self.index];
            if (std.ascii.isWhitespace(byte) or byte == ',' or byte == ']' or byte == '}' or byte == ')' or byte == ':') break;
            self.index += 1;
        }
        const token = self.input[start..self.index];
        if (std.mem.eql(u8, token, "True") or std.mem.eql(u8, token, "true")) {
            try self.writer.writeAll("true");
        } else if (std.mem.eql(u8, token, "False") or std.mem.eql(u8, token, "false")) {
            try self.writer.writeAll("false");
        } else if (std.mem.eql(u8, token, "None") or std.mem.eql(u8, token, "null")) {
            try self.writer.writeAll("null");
        } else {
            if (token.len == 0) return error.InvalidLocalProviderNativeToolCall;
            for (token) |byte| {
                if (!std.ascii.isDigit(byte) and byte != '-' and byte != '.' and byte != 'e' and byte != 'E') {
                    return error.InvalidLocalProviderNativeToolCall;
                }
            }
            try self.writer.writeAll(token);
        }
    }

    fn parseArray(self: *PythonValueParser) anyerror!void {
        try self.expect('[');
        try self.writer.writeByte('[');
        self.skipWhitespace();
        if (self.index < self.input.len and self.input[self.index] == ']') {
            self.index += 1;
            try self.writer.writeByte(']');
            return;
        }
        var item_index: usize = 0;
        while (true) {
            if (item_index > 0) try self.writer.writeByte(',');
            try self.parseValue();
            item_index += 1;
            self.skipWhitespace();
            if (self.index >= self.input.len) return error.InvalidLocalProviderNativeToolCall;
            if (self.input[self.index] == ']') {
                self.index += 1;
                try self.writer.writeByte(']');
                return;
            }
            try self.expect(',');
        }
    }

    fn parseObject(self: *PythonValueParser) anyerror!void {
        try self.expect('{');
        try self.writer.writeByte('{');
        self.skipWhitespace();
        if (self.index < self.input.len and self.input[self.index] == '}') {
            self.index += 1;
            try self.writer.writeByte('}');
            return;
        }
        var item_index: usize = 0;
        while (true) {
            if (item_index > 0) try self.writer.writeByte(',');
            self.skipWhitespace();
            if (self.index < self.input.len and (self.input[self.index] == '\'' or self.input[self.index] == '"')) {
                try self.parseString();
            } else {
                const key = try self.parseIdentifier();
                try std.json.Stringify.value(key, .{}, self.writer);
            }
            try self.expect(':');
            try self.writer.writeByte(':');
            try self.parseValue();
            item_index += 1;
            self.skipWhitespace();
            if (self.index >= self.input.len) return error.InvalidLocalProviderNativeToolCall;
            if (self.input[self.index] == '}') {
                self.index += 1;
                try self.writer.writeByte('}');
                return;
            }
            try self.expect(',');
        }
    }

    fn parseValue(self: *PythonValueParser) anyerror!void {
        self.skipWhitespace();
        if (self.index >= self.input.len) return error.InvalidLocalProviderNativeToolCall;
        switch (self.input[self.index]) {
            '\'', '"' => try self.parseString(),
            '[' => try self.parseArray(),
            '{' => try self.parseObject(),
            else => try self.parseBare(),
        }
    }

    fn parseDocument(self: *PythonValueParser) !void {
        try self.parseValue();
        self.skipWhitespace();
        if (self.index != self.input.len) return error.InvalidLocalProviderNativeToolCall;
    }
};

fn toolIsAvailable(selection: stream_provider.ToolSelection, name: []const u8) bool {
    for (selection.additional_functions) |tool| if (std.mem.eql(u8, tool.name, name)) return true;
    for (selection.selected_dynamic) |tool| if (std.mem.eql(u8, tool.name, name)) return true;
    for (selection.advertised_names) |advertised| {
        if (std.mem.eql(u8, advertised, name) and selection.advertisedFunction(name) != null) return true;
    }
    return false;
}

fn appendFallbackToolCall(
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    calls: *std.ArrayList(types.ToolCall),
    name: []const u8,
    arguments_json: []u8,
) !void {
    errdefer alloc.free(arguments_json);
    if (name.len == 0 or name.len > max_tool_identity_bytes or !toolIsAvailable(selection, name)) {
        return error.InvalidLocalProviderNativeToolCall;
    }
    if (arguments_json.len > max_tool_arguments_bytes or
        try types.ToolArgumentIntegrity.classifySerialized(alloc, arguments_json) == .malformed_json)
    {
        return error.InvalidLocalProviderNativeToolCall;
    }
    const id = try std.fmt.allocPrint(alloc, "local_call_{d}", .{calls.items.len});
    errdefer alloc.free(id);
    const owned_name = try alloc.dupe(u8, name);
    errdefer alloc.free(owned_name);
    try calls.append(alloc, .{
        .id = id,
        .name = owned_name,
        .arguments_json = arguments_json,
        .argument_integrity = .valid,
    });
}

fn freeFallbackCalls(alloc: Allocator, calls: *std.ArrayList(types.ToolCall)) void {
    for (calls.items) |call| types.freeToolCall(alloc, call);
    calls.clearRetainingCapacity();
}

const PythonCallParser = struct {
    input: []const u8,
    index: usize = 0,
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    calls: *std.ArrayList(types.ToolCall),

    fn skipWhitespace(self: *PythonCallParser) void {
        while (self.index < self.input.len and std.ascii.isWhitespace(self.input[self.index])) : (self.index += 1) {}
    }

    fn expect(self: *PythonCallParser, byte: u8) !void {
        self.skipWhitespace();
        if (self.index >= self.input.len or self.input[self.index] != byte) return error.InvalidLocalProviderNativeToolCall;
        self.index += 1;
    }

    fn parseIdentifier(self: *PythonCallParser) ![]const u8 {
        self.skipWhitespace();
        if (self.index >= self.input.len or !isIdentifierStart(self.input[self.index])) return error.InvalidLocalProviderNativeToolCall;
        const start = self.index;
        self.index += 1;
        while (self.index < self.input.len and isIdentifierContinue(self.input[self.index])) : (self.index += 1) {}
        return self.input[start..self.index];
    }

    fn parseCall(self: *PythonCallParser) !void {
        const name = try self.parseIdentifier();
        try self.expect('(');

        var output: std.Io.Writer.Allocating = .init(self.alloc);
        defer output.deinit();
        try output.writer.writeByte('{');
        self.skipWhitespace();
        var argument_index: usize = 0;
        if (self.index < self.input.len and self.input[self.index] != ')') while (true) {
            const key = try self.parseIdentifier();
            try self.expect('=');
            if (argument_index > 0) try output.writer.writeByte(',');
            try std.json.Stringify.value(key, .{}, &output.writer);
            try output.writer.writeByte(':');

            var value_parser = PythonValueParser{
                .input = self.input,
                .index = self.index,
                .alloc = self.alloc,
                .writer = &output.writer,
            };
            try value_parser.parseValue();
            self.index = value_parser.index;
            argument_index += 1;
            self.skipWhitespace();
            if (self.index < self.input.len and self.input[self.index] == ')') break;
            try self.expect(',');
        };
        try self.expect(')');
        try output.writer.writeByte('}');
        const arguments_json = try output.toOwnedSlice();
        try appendFallbackToolCall(self.alloc, self.selection, self.calls, name, arguments_json);
    }

    fn parseDocument(self: *PythonCallParser) !void {
        try self.expect('[');
        self.skipWhitespace();
        if (self.index >= self.input.len or self.input[self.index] == ']') return error.InvalidLocalProviderNativeToolCall;
        while (true) {
            try self.parseCall();
            self.skipWhitespace();
            if (self.index >= self.input.len) return error.InvalidLocalProviderNativeToolCall;
            if (self.input[self.index] == ']') {
                self.index += 1;
                break;
            }
            try self.expect(',');
        }
        self.skipWhitespace();
        if (self.index != self.input.len) return error.InvalidLocalProviderNativeToolCall;
    }
};

fn appendJsonFallbackCall(
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    calls: *std.ArrayList(types.ToolCall),
    raw: std.json.Value,
) !void {
    if (raw != .object) return error.InvalidLocalProviderNativeToolCall;
    const source = if (raw.object.get("function")) |function| function else raw;
    if (source != .object) return error.InvalidLocalProviderNativeToolCall;
    const name = source.object.get("name") orelse return error.InvalidLocalProviderNativeToolCall;
    if (name != .string) return error.InvalidLocalProviderNativeToolCall;
    const raw_arguments = source.object.get("arguments") orelse source.object.get("parameters");
    const arguments_json = if (raw_arguments) |arguments| switch (arguments) {
        .string => |serialized| blk: {
            const copy = try alloc.dupe(u8, serialized);
            if (try types.ToolArgumentIntegrity.classifySerialized(alloc, copy) == .malformed_json) {
                alloc.free(copy);
                return error.InvalidLocalProviderNativeToolCall;
            }
            break :blk copy;
        },
        .object, .array => blk: {
            var output: std.Io.Writer.Allocating = .init(alloc);
            defer output.deinit();
            try std.json.Stringify.value(arguments, .{}, &output.writer);
            break :blk try output.toOwnedSlice();
        },
        else => return error.InvalidLocalProviderNativeToolCall,
    } else try alloc.dupe(u8, "{}");
    try appendFallbackToolCall(alloc, selection, calls, name.string, arguments_json);
}

fn appendJsonFallbackValue(
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    calls: *std.ArrayList(types.ToolCall),
    raw: std.json.Value,
) !void {
    switch (raw) {
        .array => |items| {
            if (items.items.len == 0) return error.InvalidLocalProviderNativeToolCall;
            for (items.items) |item| try appendJsonFallbackCall(alloc, selection, calls, item);
        },
        .object => try appendJsonFallbackCall(alloc, selection, calls, raw),
        else => return error.InvalidLocalProviderNativeToolCall,
    }
}

fn parsePythonValueJson(alloc: Allocator, payload: []const u8) ![]u8 {
    var output: std.Io.Writer.Allocating = .init(alloc);
    defer output.deinit();
    var parser = PythonValueParser{
        .input = payload,
        .alloc = alloc,
        .writer = &output.writer,
    };
    try parser.parseDocument();
    return try output.toOwnedSlice();
}

fn parseTaggedToolPayload(
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    payload: []const u8,
) !?[]types.ToolCall {
    if (payload.len == 0 or payload.len > max_tool_arguments_bytes) return null;
    var calls: std.ArrayList(types.ToolCall) = .empty;
    var transferred = false;
    defer {
        if (!transferred) freeFallbackCalls(alloc, &calls);
        calls.deinit(alloc);
    }

    var native_parser = PythonCallParser{
        .input = payload,
        .alloc = alloc,
        .selection = selection,
        .calls = &calls,
    };
    native_parser.parseDocument() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            freeFallbackCalls(alloc, &calls);
        },
    };
    if (calls.items.len > 0) {
        const owned = try calls.toOwnedSlice(alloc);
        transferred = true;
        return owned;
    }

    const json_text = parsePythonValueJson(alloc, payload) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer alloc.free(json_text);
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer parsed.deinit();
    appendJsonFallbackValue(alloc, selection, &calls, parsed.value) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            freeFallbackCalls(alloc, &calls);
            return null;
        },
    };
    if (calls.items.len == 0) return null;
    const owned = try calls.toOwnedSlice(alloc);
    transferred = true;
    return owned;
}

fn parseTaggedToolCalls(
    alloc: Allocator,
    selection: stream_provider.ToolSelection,
    content: []const u8,
) !?[]types.ToolCall {
    var calls: std.ArrayList(types.ToolCall) = .empty;
    var transferred = false;
    defer {
        if (!transferred) freeFallbackCalls(alloc, &calls);
        calls.deinit(alloc);
    }
    var offset: usize = 0;
    while (nextTaggedToolCall(content, offset)) |tag| {
        const payload_start = tag.index + tag.open_len;
        const close_relative = std.mem.indexOf(u8, content[payload_start..], tag.close) orelse break;
        const close_start = payload_start + close_relative;
        const payload = std.mem.trim(u8, content[payload_start..close_start], " \t\r\n");
        if (try parseTaggedToolPayload(alloc, selection, payload)) |parsed| {
            errdefer types.freeToolCallSlice(alloc, parsed);
            try calls.appendSlice(alloc, parsed);
            alloc.free(parsed);
        }
        offset = close_start + tag.close.len;
    }
    if (calls.items.len == 0) return null;
    const owned = try calls.toOwnedSlice(alloc);
    transferred = true;
    return owned;
}

fn stripTaggedToolCallMarkup(content: []u8) usize {
    var read: usize = 0;
    var write: usize = 0;
    while (nextTaggedToolCall(content, read)) |tag| {
        const payload_start = tag.index + tag.open_len;
        const close_relative = std.mem.indexOf(u8, content[payload_start..], tag.close) orelse break;
        const copy_len = tag.index - read;
        if (copy_len > 0) {
            std.mem.copyForwards(u8, content[write .. write + copy_len], content[read..tag.index]);
            write += copy_len;
        }
        read = payload_start + close_relative + tag.close.len;
    }
    if (read < content.len) {
        const copy_len = content.len - read;
        std.mem.copyForwards(u8, content[write .. write + copy_len], content[read..]);
        write += copy_len;
    }
    return write;
}

const Reducer = struct {
    events: stream_provider.EventSink,
    available_tools: stream_provider.ToolSelection = .{},
    content: std.ArrayList(u8) = .empty,
    pending_content: std.ArrayList(u8) = .empty,
    fallback_candidate: bool = false,
    tools: std.ArrayList(ToolAccumulator) = .empty,
    generation_id: ?[]u8 = null,
    finish_reason: ?types.ProviderFinishReason = null,
    usage: types.Usage = .{},
    event_count: usize = 0,
    aggregate_bytes: usize = 0,

    fn init(events: stream_provider.EventSink, available_tools: stream_provider.ToolSelection) Reducer {
        return .{ .events = events, .available_tools = available_tools };
    }

    fn deinit(self: *Reducer, alloc: Allocator) void {
        self.content.deinit(alloc);
        self.pending_content.deinit(alloc);
        for (self.tools.items) |*tool| tool.deinit(alloc);
        self.tools.deinit(alloc);
        if (self.generation_id) |id| alloc.free(id);
        self.* = undefined;
    }

    fn checkLimits(self: *Reducer, json_len: usize) !void {
        self.event_count += 1;
        if (self.event_count > max_sse_events) return error.LocalProviderEventLimitExceeded;
        self.aggregate_bytes = std.math.add(usize, self.aggregate_bytes, json_len) catch
            return error.LocalProviderResponseTooLarge;
        if (self.aggregate_bytes > max_sse_aggregate_bytes) return error.LocalProviderResponseTooLarge;
    }

    fn toolAt(self: *Reducer, alloc: Allocator, index: usize) !*ToolAccumulator {
        if (index >= max_tool_calls) return error.LocalProviderToolCallLimitExceeded;
        if (index != self.tools.items.len) {
            if (index < self.tools.items.len) return &self.tools.items[index];
            return error.LocalProviderToolCallOrderInvalid;
        }
        try self.tools.append(alloc, .{});
        return &self.tools.items[index];
    }

    fn setToolText(alloc: Allocator, target: *[]u8, value: []const u8, limit: usize) !void {
        if (value.len == 0) return;
        if (value.len > limit) return error.LocalProviderToolIdentityTooLarge;
        if (target.*.len == 0) {
            target.* = try alloc.dupe(u8, value);
            return;
        }
        if (!std.mem.eql(u8, target.*, value)) return error.LocalProviderToolIdentityConflict;
    }

    fn emitToolStart(self: *Reducer, tool: *ToolAccumulator) void {
        if (tool.started or tool.id.len == 0 or tool.name.len == 0) return;
        tool.started = true;
        self.events.emit(.{ .tool_started = .{
            .id = tool.id,
            .name = tool.name,
        } });
    }

    fn emitPendingContent(self: *Reducer, keep: usize) void {
        const emit_len = self.pending_content.items.len - keep;
        if (emit_len == 0) return;
        self.events.emit(.{ .content_delta = self.pending_content.items[0..emit_len] });
        const remaining = self.pending_content.items.len - emit_len;
        if (remaining > 0) {
            std.mem.copyForwards(
                u8,
                self.pending_content.items[0..remaining],
                self.pending_content.items[emit_len..],
            );
        }
        self.pending_content.items.len = remaining;
    }

    fn partialFallbackOpenLength(content: []const u8) usize {
        var longest: usize = 0;
        for (fallback_tool_tags) |tag| {
            const max_len = @min(content.len, tag.open.len - 1);
            var len = max_len;
            while (len > longest) : (len -= 1) {
                if (len > 0 and std.mem.eql(u8, content[content.len - len ..], tag.open[0..len])) {
                    longest = len;
                    break;
                }
            }
        }
        return longest;
    }

    fn flushPendingContent(self: *Reducer) void {
        if (self.fallback_candidate) return;
        if (nextTaggedToolCall(self.pending_content.items, 0)) |tag| {
            self.fallback_candidate = true;
            self.emitPendingContent(self.pending_content.items.len - tag.index);
            return;
        }
        self.emitPendingContent(partialFallbackOpenLength(self.pending_content.items));
    }

    fn appendContent(self: *Reducer, alloc: Allocator, value: []const u8, capture_limit: ?usize) !void {
        if (value.len == 0) return;
        const remaining = if (capture_limit) |limit|
            limit -| self.content.items.len
        else
            value.len;
        if (remaining > 0) try self.content.appendSlice(alloc, value[0..@min(remaining, value.len)]);
        try self.pending_content.appendSlice(alloc, value);
        self.flushPendingContent();
    }

    fn apply(
        self: *Reducer,
        alloc: Allocator,
        json_text: []const u8,
        cancel_flag: *std.atomic.Value(bool),
        capture_limit: ?usize,
    ) !void {
        if (cancel_flag.load(.seq_cst)) return error.Cancelled;
        try self.checkLimits(json_text.len);
        var parsed = std.json.parseFromSlice(std.json.Value, alloc, json_text, .{}) catch
            return error.InvalidLocalProviderEvent;
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidLocalProviderEvent;
        if (parsed.value.object.get("error")) |_| return error.LocalProviderResponseFailed;

        if (parsed.value.object.get("id")) |id| if (id == .string and self.generation_id == null) {
            self.generation_id = try alloc.dupe(u8, id.string);
        };
        if (parsed.value.object.get("usage")) |usage| if (usage == .object) {
            if (usage.object.get("prompt_tokens")) |value| {
                if (value == .integer and value.integer >= 0) self.usage.input_tokens = @intCast(value.integer);
            }
            if (usage.object.get("completion_tokens")) |value| {
                if (value == .integer and value.integer >= 0) self.usage.output_tokens = @intCast(value.integer);
            }
            if (usage.object.get("prompt_tokens_details")) |details| if (details == .object) {
                if (details.object.get("cached_tokens")) |value| {
                    if (value == .integer and value.integer >= 0) self.usage.cache_read_tokens = @intCast(value.integer);
                }
            };
            if (usage.object.get("completion_tokens_details")) |details| if (details == .object) {
                if (details.object.get("reasoning_tokens")) |value| {
                    if (value == .integer and value.integer >= 0) self.usage.reasoning_tokens = @intCast(value.integer);
                }
            };
        };

        const choices = parsed.value.object.get("choices") orelse return;
        if (choices != .array) return error.InvalidLocalProviderEvent;
        if (choices.array.items.len == 0) return;
        const choice = choices.array.items[0];
        if (choice != .object) return error.InvalidLocalProviderEvent;
        if (choice.object.get("finish_reason")) |reason| if (reason == .string and reason.string.len > 0) {
            self.finish_reason = types.ProviderFinishReason.parse_legacy(reason.string) orelse .other;
        };
        const delta = choice.object.get("delta") orelse return;
        if (delta != .object) return error.InvalidLocalProviderEvent;
        if (delta.object.get("content")) |content| if (content == .string) {
            try self.appendContent(alloc, content.string, capture_limit);
        };
        const reasoning = delta.object.get("reasoning_content") orelse delta.object.get("reasoning");
        if (reasoning) |value| if (value == .string and value.string.len > 0) {
            self.events.emit(.{ .reasoning_delta = value.string });
        };
        if (delta.object.get("tool_calls")) |tool_calls| if (tool_calls == .array) {
            for (tool_calls.array.items, 0..) |raw_tool, position| {
                if (raw_tool != .object) return error.InvalidLocalProviderEvent;
                const index = if (raw_tool.object.get("index")) |value|
                    if (value == .integer and value.integer >= 0) @as(usize, @intCast(value.integer)) else return error.InvalidLocalProviderEvent
                else
                    position;
                const tool = try self.toolAt(alloc, index);
                if (raw_tool.object.get("id")) |id| if (id == .string) try setToolText(alloc, &tool.id, id.string, max_tool_identity_bytes);
                const function = raw_tool.object.get("function") orelse return error.InvalidLocalProviderEvent;
                if (function != .object) return error.InvalidLocalProviderEvent;
                if (function.object.get("name")) |name| if (name == .string) try setToolText(alloc, &tool.name, name.string, max_tool_identity_bytes);
                if (function.object.get("arguments")) |arguments| if (arguments == .string) {
                    if (std.math.add(usize, tool.arguments.items.len, arguments.string.len) catch max_tool_arguments_bytes > max_tool_arguments_bytes) {
                        return error.LocalProviderToolArgumentsTooLarge;
                    }
                    try tool.arguments.appendSlice(alloc, arguments.string);
                    self.events.emit(.{ .tool_input_delta = arguments.string });
                };
                self.emitToolStart(tool);
            }
        };
    }

    fn finish(self: *Reducer, alloc: Allocator) !types.ModelCompletion {
        if (self.finish_reason == null) self.finish_reason = .stop;
        if (self.tools.items.len == 0) {
            if (try parseTaggedToolCalls(alloc, self.available_tools, self.content.items)) |fallback_calls| {
                self.finish_reason = .tool_calls;
                self.content.items.len = stripTaggedToolCallMarkup(self.content.items);
                const pending_len = stripTaggedToolCallMarkup(self.pending_content.items);
                if (pending_len > 0) {
                    self.events.emit(.{ .content_delta = self.pending_content.items[0..pending_len] });
                }
                self.pending_content.items.len = 0;
                return self.finishFallback(alloc, fallback_calls);
            }
        }
        self.emitPendingContent(0);
        var tool_calls = try alloc.alloc(types.ToolCall, self.tools.items.len);
        var initialized: usize = 0;
        errdefer {
            for (tool_calls[0..initialized]) |call| {
                alloc.free(call.id);
                alloc.free(call.name);
                alloc.free(call.arguments_json);
            }
            if (tool_calls.len > 0) alloc.free(tool_calls);
        }
        for (self.tools.items, 0..) |*tool, index| {
            if (tool.id.len == 0) tool.id = try std.fmt.allocPrint(alloc, "local_call_{d}", .{index});
            if (tool.name.len == 0) return error.InvalidLocalProviderToolCall;
            self.emitToolStart(tool);
            const arguments = if (tool.arguments.items.len == 0) "{}" else tool.arguments.items;
            if (arguments.len > max_tool_arguments_bytes) return error.LocalProviderToolArgumentsTooLarge;
            if (try types.ToolArgumentIntegrity.classifySerialized(alloc, arguments) == .malformed_json) {
                return error.InvalidLocalProviderToolArguments;
            }
            tool_calls[initialized] = .{
                .id = try alloc.dupe(u8, tool.id),
                .name = try alloc.dupe(u8, tool.name),
                .arguments_json = try alloc.dupe(u8, arguments),
            };
            initialized += 1;
        }
        if (initialized == 0) {
            alloc.free(tool_calls);
            tool_calls = &.{};
        }
        const content = if (self.content.items.len > 0) try self.content.toOwnedSlice(alloc) else null;
        const completion = types.ModelCompletion{
            .content = content,
            .tool_calls = tool_calls,
            .generation_id = self.generation_id,
            .finish_reason = self.finish_reason,
            .usage = self.usage,
        };
        self.generation_id = null;
        return completion;
    }

    fn finishFallback(self: *Reducer, alloc: Allocator, tool_calls: []types.ToolCall) !types.ModelCompletion {
        errdefer types.freeToolCallSlice(alloc, tool_calls);
        const content = if (self.content.items.len > 0) try self.content.toOwnedSlice(alloc) else null;
        const completion = types.ModelCompletion{
            .content = content,
            .tool_calls = tool_calls,
            .generation_id = self.generation_id,
            .finish_reason = self.finish_reason,
            .usage = self.usage,
        };
        self.generation_id = null;
        return completion;
    }
};

const SseReader = struct {
    pending_line: std.ArrayList(u8) = .empty,

    fn deinit(self: *SseReader, alloc: Allocator) void {
        self.pending_line.deinit(alloc);
    }

    fn release(self: *SseReader) void {
        self.pending_line.clearRetainingCapacity();
    }

    fn next(self: *SseReader, alloc: Allocator, reader: anytype) !?[]const u8 {
        while (true) {
            const line = try self.readLine(alloc, reader) orelse return null;
            const trimmed = std.mem.trim(u8, line, " \t\r");
            if (trimmed.len == 0 or trimmed[0] == ':') {
                self.release();
                continue;
            }
            if (!std.mem.startsWith(u8, trimmed, "data:")) {
                self.release();
                continue;
            }
            const data = std.mem.trim(u8, trimmed["data:".len..], " \t");
            if (std.mem.eql(u8, data, "[DONE]")) return null;
            return data;
        }
    }

    fn readLine(self: *SseReader, alloc: Allocator, reader: anytype) !?[]const u8 {
        while (true) {
            const fragment = reader.takeDelimiter('\n') catch |err| switch (err) {
                error.StreamTooLong => {
                    const buffered = reader.buffered();
                    if (buffered.len == 0) return error.LocalProviderSseReadStalled;
                    if (buffered.len > max_sse_line_bytes - self.pending_line.items.len) return error.LocalProviderSseEventTooLarge;
                    try self.pending_line.appendSlice(alloc, buffered);
                    reader.tossBuffered();
                    continue;
                },
                error.ReadFailed => return error.ReadFailed,
            } orelse {
                if (self.pending_line.items.len > 0) return self.pending_line.items;
                return null;
            };
            if (fragment.len > max_sse_line_bytes - self.pending_line.items.len) return error.LocalProviderSseEventTooLarge;
            if (self.pending_line.items.len == 0) return fragment;
            try self.pending_line.appendSlice(alloc, fragment);
            return self.pending_line.items;
        }
    }
};

fn consumeSse(
    alloc: Allocator,
    reader: anytype,
    request: stream_provider.ModelRequest,
) !types.ModelCompletion {
    var reducer = Reducer.init(request.events, request.tools);
    defer reducer.deinit(alloc);
    var sse: SseReader = .{};
    defer sse.deinit(alloc);
    while (try sse.next(alloc, reader)) |json_text| {
        defer sse.release();
        try reducer.apply(alloc, json_text, request.cancel_flag, request.content_capture_limit);
    }
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    return reducer.finish(alloc);
}

fn failureKind(status: std.http.Status) stream_provider.FailureKind {
    return switch (status) {
        .bad_request => .invalid_request,
        .unauthorized => .unauthorized,
        .forbidden => .forbidden,
        .payload_too_large => .request_too_large,
        .too_many_requests => .rate_limited,
        .internal_server_error => .server_error,
        .bad_gateway => .bad_gateway,
        .service_unavailable => .unavailable,
        .gateway_timeout => .gateway_timeout,
        else => .provider_error,
    };
}

const OpenedRequest = struct {
    request: ?std.http.Client.Request,

    pub fn deinit(self: *OpenedRequest, _: Allocator) void {
        if (self.request) |*request| request.deinit();
        self.request = null;
    }

    pub fn take(self: *OpenedRequest) std.http.Client.Request {
        const request = self.request.?;
        self.request = null;
        return request;
    }
};

const OpenRequestOperation = struct {
    client: *std.http.Client,
    uri: std.Uri,
    api_key: ?[]const u8,

    pub fn run(self: *@This()) !OpenedRequest {
        var headers: std.http.Client.Request.Headers = .{
            .content_type = .{ .override = "application/json" },
            .accept_encoding = .omit,
            .user_agent = .{ .override = gateway_client.user_agent },
        };
        if (self.api_key) |api_key| headers.authorization = .{ .override = api_key };
        return .{ .request = try self.client.request(.POST, self.uri, .{
            .headers = headers,
            .extra_headers = &.{.{ .name = "accept", .value = "text/event-stream" }},
            .keep_alive = false,
            .redirect_behavior = .unhandled,
        }) };
    }
};

fn streamCompletion(
    _: ?*anyopaque,
    alloc: Allocator,
    request: stream_provider.ModelRequest,
) !stream_provider.Result {
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
    try validateModel(request.model);
    const payload = try buildRequest(alloc, request.data());
    defer alloc.free(payload);
    var result = streamPrepared(alloc, request, payload) catch |err| {
        if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;
        if (requestDeadlineExpired(request)) return error.Timeout;
        request.attempt_evidence.network_failure = gateway_client.networkFailureEvidence(err, request.delivery.load());
        return err;
    };
    if (requestDeadlineExpired(request)) {
        result.deinit(alloc);
        return error.Timeout;
    }
    return result;
}

fn requestDeadlineExpired(request: stream_provider.ModelRequest) bool {
    const deadline = request.deadline orelse return false;
    const now = std.Io.Clock.Timestamp.now(io_mod.getIo(), .awake);
    return !std.Io.Clock.Timestamp.compare(now, .lt, deadline);
}

fn isAllowedChatUrl(url: []const u8, allow_non_loopback: bool) bool {
    if (gateway_client.isLoopbackHttpUrl(url)) return true;
    if (!allow_non_loopback) return false;

    const uri = std.Uri.parse(url) catch return false;
    return std.ascii.eqlIgnoreCase(uri.scheme, "http") and
        uri.user == null and
        uri.password == null and
        uri.host != null and
        uri.port != null;
}

pub fn streamPrepared(
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,
) !stream_provider.Result {
    const chat_url = io_mod.getenv(chat_url_env) orelse default_chat_url;
    const allow_non_loopback = std.mem.eql(u8, io_mod.getenv(allow_non_loopback_env) orelse "", "1");
    if (!isAllowedChatUrl(chat_url, allow_non_loopback)) return error.InvalidLocalChatUrl;
    const uri = try std.Uri.parse(chat_url);
    const api_key = io_mod.getenv(api_key_env);
    const auth_header = if (api_key) |value| if (value.len > 0)
        try std.fmt.allocPrint(alloc, "Bearer {s}", .{value})
    else
        null else null;
    defer if (auth_header) |value| alloc.free(value);

    var client: std.http.Client = .{ .allocator = alloc, .io = io_mod.getIo() };
    defer client.deinit();
    var open_operation = OpenRequestOperation{
        .client = &client,
        .uri = uri,
        .api_key = auth_header,
    };
    var connect_deadline = std.Io.Clock.Timestamp.fromNow(io_mod.getIo(), .{
        .clock = .awake,
        .raw = .fromMilliseconds(connect_timeout_ms),
    });
    if (request.deadline) |deadline| {
        if (std.Io.Clock.Timestamp.compare(deadline, .lt, connect_deadline)) connect_deadline = deadline;
    }
    try request.admission.admit();
    var opened = try gateway_client.runBoundedHttpOperation(
        OpenedRequest,
        alloc,
        request.cancel_flag,
        connect_deadline,
        &open_operation,
    );
    var http_request = opened.take();
    defer http_request.deinit();
    var cancel_watch_done = std.atomic.Value(bool).init(false);
    const cancel_watcher = if (http_request.connection) |connection|
        if (request.deadline) |deadline|
            try gateway_client.spawnHttpCancelWatcherBounded(
                &cancel_watch_done,
                request.cancel_flag,
                deadline,
                connection.stream_writer.stream,
            )
        else
            try gateway_client.spawnHttpCancelWatcher(
                &cancel_watch_done,
                request.cancel_flag,
                connection.stream_writer.stream,
            )
    else
        null;
    defer {
        cancel_watch_done.store(true, .seq_cst);
        if (cancel_watcher) |thread| thread.join();
    }
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    http_request.transfer_encoding = .{ .content_length = payload.len };
    var send_buffer: [8192]u8 = undefined;
    request.delivery.markPossiblySent();
    var body_writer = try http_request.sendBodyUnflushed(&send_buffer);
    try body_writer.writer.writeAll(payload);
    try body_writer.end();
    if (http_request.connection) |connection| try connection.flush();
    if (request.cancel_flag.load(.seq_cst)) return error.Cancelled;

    var response = try http_request.receiveHead(&.{});
    if (response.head.status != .ok) {
        var transfer: [16 * 1024]u8 = undefined;
        const reader = response.reader(&transfer);
        const body = reader.allocRemaining(alloc, .limited(max_error_body_bytes)) catch |err| switch (err) {
            error.StreamTooLong => try alloc.dupe(u8, "local provider error response exceeded the local limit"),
            else => return err,
        };
        return .{ .failed = .{
            .kind = failureKind(response.head.status),
            .detail = body,
            .ownership = .owned,
        } };
    }

    var transfer_buffer: [transfer_buffer_bytes]u8 = undefined;
    const reader = response.reader(&transfer_buffer);
    const completion = try consumeSse(alloc, reader, request);
    return .{ .completed = .{
        .completion = completion,
        .usage = .{ .unavailable = .unbilled },
        .ownership = .owned,
    } };
}

test "local OpenAI-compatible request uses Chat Completions tool schema" {
    const alloc = std.testing.allocator;
    const function = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read file",
        .input_schema = .{},
    };
    const messages = [_]types.ChatMessage{.{ .role = .user, .content = "Read it." }};
    const body = try buildRequest(alloc, .{
        .model = "lfm2.5-8b-a1b",
        .messages = &messages,
        .tools = .{ .additional_functions = &.{function} },
        .tool_choice = .auto,
        .provider_options = .{},
    });
    defer alloc.free(body);
    try std.testing.expect(std.mem.find(u8, body, "\"chat.completions\"") == null);
    try std.testing.expect(std.mem.find(u8, body, "\"tools\":[{\"type\":\"function\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"parameters\":{\"type\":\"object\"") != null);
}

test "local OpenAI-compatible request rejects images" {
    const images = [_]types.ImageAttachment{.{ .path = @constCast("image.png"), .media_type = @constCast("image/png") }};
    try std.testing.expectError(
        error.LocalProviderVisionUnsupported,
        buildRequest(std.testing.allocator, .{
            .model = "local",
            .messages = &.{.{ .role = .user, .content = "look", .images = &images }},
            .tool_choice = .none,
            .provider_options = .{},
        }),
    );
}

test "local OpenAI-compatible URL stays loopback unless explicitly enabled" {
    try std.testing.expect(isAllowedChatUrl("http://127.0.0.1:1234/v1/chat/completions", false));
    try std.testing.expect(!isAllowedChatUrl("http://172.29.128.1:1234/v1/chat/completions", false));
    try std.testing.expect(isAllowedChatUrl("http://172.29.128.1:1234/v1/chat/completions", true));
    try std.testing.expect(!isAllowedChatUrl("https://172.29.128.1:1234/v1/chat/completions", true));
}

test "local OpenAI-compatible reducer assembles streamed tool arguments" {
    const Capture = struct {
        fn emit(_: *anyopaque, _: stream_provider.Event) void {}
    };
    const function = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read file",
        .input_schema = .{},
    };
    var capture: u8 = 0;
    var reducer = Reducer.init(
        .{ .context = &capture, .emit_fn = Capture.emit },
        .{ .additional_functions = &.{function} },
    );
    defer reducer.deinit(std.testing.allocator);
    var cancelled = std.atomic.Value(bool).init(false);

    try reducer.apply(
        std.testing.allocator,
        "{\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"src/\"}}]}}]}",
        &cancelled,
        null,
    );
    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"main.zig\\\"}\"}}]},\"finish_reason\":\"tool_calls\"}]}",
        &cancelled,
        null,
    );

    const completion = try reducer.finish(std.testing.allocator);
    defer {
        if (completion.content) |content| std.testing.allocator.free(@constCast(content));
        if (completion.generation_id) |id| std.testing.allocator.free(@constCast(id));
        types.freeToolCallSlice(std.testing.allocator, @constCast(completion.tool_calls));
    }
    try std.testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try std.testing.expectEqualStrings("call_1", completion.tool_calls[0].id);
    try std.testing.expectEqualStrings("read_file", completion.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"src/main.zig\"}", completion.tool_calls[0].arguments_json);
    try std.testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
    try std.testing.expectEqualStrings("chatcmpl-1", completion.generation_id.?);
}

test "local OpenAI-compatible reducer parses native Liquid Pythonic calls" {
    const Capture = struct {
        fn emit(_: *anyopaque, _: stream_provider.Event) void {}
    };
    const function = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read file",
        .input_schema = .{},
    };
    var capture: u8 = 0;
    var reducer = Reducer.init(
        .{ .context = &capture, .emit_fn = Capture.emit },
        .{ .additional_functions = &.{function} },
    );
    defer reducer.deinit(std.testing.allocator);
    var cancelled = std.atomic.Value(bool).init(false);

    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{\"content\":\"<|tool_call_start|>[read_file(path=\\\"README.md\\\")]<|tool_call_end|>after\"}}]}",
        &cancelled,
        null,
    );
    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
        &cancelled,
        null,
    );

    const completion = try reducer.finish(std.testing.allocator);
    defer {
        if (completion.content) |content| std.testing.allocator.free(@constCast(content));
        if (completion.generation_id) |id| std.testing.allocator.free(@constCast(id));
        types.freeToolCallSlice(std.testing.allocator, @constCast(completion.tool_calls));
    }
    try std.testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try std.testing.expectEqualStrings("local_call_0", completion.tool_calls[0].id);
    try std.testing.expectEqualStrings("read_file", completion.tool_calls[0].name);
    try std.testing.expectEqualStrings("{\"path\":\"README.md\"}", completion.tool_calls[0].arguments_json);
    try std.testing.expectEqualStrings("after", completion.content.?);
    try std.testing.expectEqual(types.ProviderFinishReason.tool_calls, completion.finish_reason.?);
}

test "local OpenAI-compatible reducer hides split fallback tool markup while streaming" {
    const Capture = struct {
        fn emit(ctx: *anyopaque, event: stream_provider.Event) void {
            switch (event) {
                .content_delta => |value| {
                    const output: *std.ArrayList(u8) = @ptrCast(@alignCast(ctx));
                    output.appendSlice(std.testing.allocator, value) catch unreachable;
                },
                else => {},
            }
        }
    };
    const function = model_tool_schema.FunctionSchema{
        .name = "read_file",
        .description = "Read file",
        .input_schema = .{},
    };
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(std.testing.allocator);
    var reducer = Reducer.init(
        .{ .context = &streamed, .emit_fn = Capture.emit },
        .{ .additional_functions = &.{function} },
    );
    defer reducer.deinit(std.testing.allocator);
    var cancelled = std.atomic.Value(bool).init(false);

    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{\"content\":\"before<|tool_call_\"}}]}",
        &cancelled,
        null,
    );
    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{\"content\":\"start|>[read_file(path=\\\"README.md\\\")]<|tool_call_end|>after\"}}]}",
        &cancelled,
        null,
    );
    try std.testing.expectEqualStrings("before", streamed.items);

    try reducer.apply(
        std.testing.allocator,
        "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
        &cancelled,
        null,
    );
    const completion = try reducer.finish(std.testing.allocator);
    defer {
        if (completion.content) |content| std.testing.allocator.free(@constCast(content));
        if (completion.generation_id) |id| std.testing.allocator.free(@constCast(id));
        types.freeToolCallSlice(std.testing.allocator, @constCast(completion.tool_calls));
    }
    try std.testing.expectEqualStrings("beforeafter", streamed.items);
    try std.testing.expect(std.mem.indexOf(u8, streamed.items, "tool_call_start") == null);
    try std.testing.expectEqual(@as(usize, 1), completion.tool_calls.len);
    try std.testing.expectEqualStrings("beforeafter", completion.content.?);
}

test "local OpenAI-compatible reducer parses tagged Python dictionaries" {
    const function = model_tool_schema.FunctionSchema{
        .name = "mcp_graphify_query_graph",
        .description = "Query Graphify",
        .input_schema = .{},
    };
    const parsed = try parseTaggedToolCalls(
        std.testing.allocator,
        .{ .additional_functions = &.{function} },
        "<tool_call>\n{'name': 'mcp_graphify_query_graph', 'arguments': {'query': '/find routing related code in the repository'}}\n</tool_call>",
    );
    try std.testing.expect(parsed != null);
    const calls = parsed.?;
    defer types.freeToolCallSlice(std.testing.allocator, calls);
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("mcp_graphify_query_graph", calls[0].name);
    try std.testing.expectEqualStrings(
        "{\"query\":\"/find routing related code in the repository\"}",
        calls[0].arguments_json,
    );
}

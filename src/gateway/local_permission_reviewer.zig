const std = @import("std");
const permission_auto_classifier = @import("../core/permissions/auto_classifier.zig");
const stream_provider = @import("../core/agent/stream_provider.zig");
const types = @import("../core/shared/types.zig");
const openai_compatible = @import("openai_compatible.zig");
const responses_reviewer = @import("responses_permission_reviewer.zig");

const Allocator = std.mem.Allocator;

pub const provider = permission_auto_classifier.Provider{
    .review_fn = reviewLocal,
};

fn reviewLocal(
    _: ?*anyopaque,
    alloc: Allocator,
    input: permission_auto_classifier.ProviderInput,
    request: permission_auto_classifier.ReviewRequest,
) anyerror!permission_auto_classifier.ParseOutcome {
    return responses_reviewer.review(alloc, input, request, .{
        .source = .ai_gateway_api_key,
        .model = request.review_turn.model,
        .require_credential = false,
        .validate_fn = validateLocal,
        .build_fn = openai_compatible.buildRequest,
        .send_fn = sendPrepared,
    });
}

fn validateLocal(_: Allocator, _: permission_auto_classifier.ProviderInput) !void {}

fn sendPrepared(
    alloc: Allocator,
    request: stream_provider.ModelRequest,
    payload: []const u8,
) anyerror!stream_provider.Result {
    return openai_compatible.streamPrepared(alloc, request, payload);
}

test "local reviewer builds native OpenAI-compatible review request without credentials" {
    const messages = [_]types.ChatMessage{
        .{ .role = .user, .content = "Run the requested check." },
        .{
            .role = .assistant,
            .tool_calls = &.{.{
                .id = "call_review",
                .name = "terminal",
                .arguments_json = "{\"action\":\"exec\",\"command\":\"pwd\"}",
            }},
        },
        .{ .role = .system, .content = "Review this exact action." },
    };
    var cancelled = std.atomic.Value(bool).init(false);
    const body = try responses_reviewer.buildPayloadForTest(
        std.testing.allocator,
        "lfm-fx-execution-v1",
        &messages,
        "call_review",
        std.Io.Clock.Timestamp.fromNow(@import("../core/shared/io.zig").getIo(), .{
            .clock = .awake,
            .raw = .fromSeconds(5),
        }),
        &cancelled,
        openai_compatible.buildRequest,
    );
    defer std.testing.allocator.free(body);

    try std.testing.expect(std.mem.find(u8, body, "\"model\":\"lfm-fx-execution-v1\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"tool_choice\":\"required\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"name\":\"permission_decision\"") != null);
    try std.testing.expect(std.mem.find(u8, body, "\"authorization\"") == null);
}

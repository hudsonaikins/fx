const std = @import("std");
const model_provider = @import("model_provider.zig");
const types = @import("../shared/types.zig");
const skill_invocation = @import("../skills/skill_invocation.zig");
const skill_runtime = @import("../skills/skill_runtime.zig");

const Allocator = std.mem.Allocator;

pub const graphify_query_tool = "mcp_graphify_query_graph";
const graphify_query_alias = "graphify_query_graph";

pub const guidance =
    "<local_repository_routing>\n" ++
    "For repository or coding work, Graphify retrieval is mandatory before local inspection. " ++
    "Call capability_search with the exact use case, select the exact Graphify query tool with " ++
    "mcp_select_tool, then call that Graphify tool. For configured server graphify, exact query tool " ++
    "name is " ++ graphify_query_tool ++ "; call it with {\"question\":\"...\"}. graphify is a server name, not a tool name. Do not guess MCP names. After Graphify returns, " ++
    "skip the graphify skill and use normal local tools for edits and tests. " ++
    "Use provider tool calls, never textual <tool_call> tags or Python dictionaries. " ++
    "Call local tools directly; do not use mcp_select_tool for local tools. " ++
    "For run, execute, test, build, format, lint, compile, or check requests without mutation, use terminal only with action=exec and preserve the requested command exactly; do not invent replacement commands or paths. " ++
    "Use workspace-relative paths or paths returned by tools; do not invent /mnt/c or drive-letter paths. " ++
    "</local_repository_routing>";

pub const continuation =
    "Graphify retrieval is still required. Call capability_search, select " ++
    graphify_query_tool ++ " with mcp_select_tool, then call " ++ graphify_query_tool ++
    " with {\"question\":\"your repository question\"} before answering.";

pub const inspection_continuation =
    "Graphify retrieval completed. Before answering, call one safe local inspection tool " ++
    "such as read_file, grep_files, glob_files, list_files, semantic_search, or terminal. " ++
    "Use workspace-relative paths or paths returned by the tool. Do not invent file contents.";

pub const terminal_continuation =
    "Graphify retrieval completed. Use terminal only. Call terminal with one action object: " ++
    "{\"action\":\"exec\",\"command\":\"the exact command requested by the user\",\"timeout_ms\":30000}. " ++
    "Preserve the requested command exactly; do not substitute commands, invent paths, or use start, write, read, or wait.";

const task_markers = [_][]const u8{
    "repository",
    "repo",
    "codebase",
    "coding",
    "code",
    "source",
    "file",
    "architecture",
    "function",
    "module",
    "implement",
    "fix",
    "debug",
    "test",
    "refactor",
    "development",
    "graphify",
};

const local_inspection_tools = [_][]const u8{
    "list_files",
    "glob_files",
    "grep_files",
    "semantic_search",
    "read_file",
    "file_info",
    "terminal",
};

const read_only_local_inspection_tools = [_][]const u8{
    "list_files",
    "glob_files",
    "grep_files",
    "semantic_search",
    "read_file",
    "file_info",
};

const mutating_or_execution_markers = [_][]const u8{
    "edit",
    "write",
    "create",
    "delete",
    "remove",
    "rename",
    "move",
    "copy",
    "implement",
    "fix",
    "debug",
    "refactor",
    "change",
    "modify",
    "patch",
    "run",
    "execute",
    "test",
    "build",
    "format",
    "lint",
    "compile",
    "install",
    "terminal",
    "shell",
    "command",
    "git",
    "commit",
    "branch",
    "history",
};

const read_only_directives = [_][]const u8{
    "do not edit",
    "don't edit",
    "no edits",
    "read only",
    "read-only",
    "without editing",
    "no changes",
    "do not change",
    "don't change",
};

const negative_execution_directives = [_][]const u8{
    "do not run",
    "don't run",
    "never run",
    "without running",
    "do not execute",
    "don't execute",
    "never execute",
    "without executing",
    "do not test",
    "don't test",
    "never test",
    "without testing",
    "do not build",
    "don't build",
    "never build",
    "without building",
    "do not format",
    "don't format",
    "do not lint",
    "don't lint",
    "do not compile",
    "don't compile",
    "do not check",
    "don't check",
    "do not use terminal",
    "don't use terminal",
    "never use terminal",
    "without terminal",
};

const terminal_request_markers = [_][]const u8{
    "run",
    "execute",
    "test",
    "build",
    "format",
    "lint",
    "compile",
    "check",
};

const mutation_markers = [_][]const u8{
    "edit",
    "write",
    "create",
    "delete",
    "remove",
    "rename",
    "move",
    "copy",
    "implement",
    "fix",
    "debug",
    "refactor",
    "change",
    "modify",
    "patch",
};

const discovery_tools = [_][]const u8{
    "capability_search",
    "mcp_search_tools",
    "mcp_select_tool",
    "mcp_features",
    "skill",
    "skill_search",
    "ask_user_question",
};

const preflight_tools = [_][]const u8{
    "capability_search",
    "mcp_search_tools",
    "mcp_select_tool",
};

pub fn requiresGraphify(provider: model_provider.ProviderId, prompt: []const u8) bool {
    if (provider != .local) return false;
    for (task_markers) |marker| {
        if (containsWordIgnoreCase(prompt, marker)) return true;
    }
    return containsFileHint(prompt);
}

pub fn isGraphifyRetrievalTool(name: []const u8) bool {
    return std.mem.eql(u8, name, graphify_query_tool) or
        std.mem.eql(u8, name, graphify_query_alias);
}

pub fn repairGraphifyRoutingCall(
    alloc: Allocator,
    provider: model_provider.ProviderId,
    prompt: []const u8,
    tool_messages: []const types.ChatMessage,
    call: types.ToolCall,
) !types.ToolCall {
    if (provider != .local or hasSuccessfulGraphifyRetrieval(tool_messages)) return call;

    if (std.mem.eql(u8, call.name, "mcp_features") or std.mem.eql(u8, call.name, "skill")) {
        return rewriteToolCallNameAndArguments(
            alloc,
            call,
            "mcp_select_tool",
            "{\"name\":\"mcp_graphify_query_graph\"}",
        );
    }

    if (std.mem.eql(u8, call.name, "mcp_select_tool")) {
        var parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments_json, .{}) catch return call;
        defer parsed.deinit();
        if (parsed.value != .object) return call;
        if (parsed.value.object.get("name")) |name| {
            if (name == .string and std.mem.eql(u8, name.string, graphify_query_tool)) return call;
        }
        return rewriteToolCallArguments(alloc, call, "{\"name\":\"mcp_graphify_query_graph\"}");
    }

    if (!std.mem.eql(u8, call.name, graphify_query_tool)) return call;

    var parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments_json, .{}) catch return call;
    defer parsed.deinit();
    if (parsed.value != .object) return call;
    if (parsed.value.object.get("question")) |question| {
        if (question == .string and question.string.len > 0) return call;
    }

    var output: std.Io.Writer.Allocating = .init(alloc);
    defer output.deinit();
    output.writer.writeAll("{\"question\":") catch return error.OutOfMemory;
    std.json.Stringify.value(prompt, .{}, &output.writer) catch return error.OutOfMemory;
    output.writer.writeByte('}') catch return error.OutOfMemory;
    const arguments_json = try output.toOwnedSlice();
    defer alloc.free(arguments_json);
    return rewriteToolCallArguments(alloc, call, arguments_json);
}

pub fn repairTerminalExecutionCall(
    alloc: Allocator,
    provider: model_provider.ProviderId,
    prompt: []const u8,
    call: types.ToolCall,
) !types.ToolCall {
    if (provider != .local or !isTerminalOnlyRequest(prompt) or
        !std.mem.eql(u8, call.name, "terminal")) return call;
    const command = extractExplicitCommand(prompt) orelse return call;

    var output: std.Io.Writer.Allocating = .init(alloc);
    defer output.deinit();
    output.writer.writeAll("{\"action\":\"exec\",\"command\":") catch return error.OutOfMemory;
    std.json.Stringify.value(command, .{}, &output.writer) catch return error.OutOfMemory;
    output.writer.writeAll(",\"timeout_ms\":30000}") catch return error.OutOfMemory;
    const arguments_json = try output.toOwnedSlice();
    defer alloc.free(arguments_json);
    return rewriteToolCallArguments(alloc, call, arguments_json);
}

fn rewriteToolCallArguments(
    alloc: Allocator,
    call: types.ToolCall,
    arguments_json: []const u8,
) !types.ToolCall {
    var repaired = try types.dupeToolCall(alloc, call);
    alloc.free(repaired.arguments_json);
    repaired.arguments_json = try alloc.dupe(u8, arguments_json);
    repaired.argument_integrity = .valid;
    return repaired;
}

fn rewriteToolCallNameAndArguments(
    alloc: Allocator,
    call: types.ToolCall,
    name: []const u8,
    arguments_json: []const u8,
) !types.ToolCall {
    var repaired = try types.dupeToolCall(alloc, call);
    errdefer types.freeToolCall(alloc, repaired);
    const rewritten_name = try alloc.dupe(u8, name);
    errdefer alloc.free(rewritten_name);
    const rewritten_arguments = try alloc.dupe(u8, arguments_json);
    errdefer alloc.free(rewritten_arguments);
    alloc.free(repaired.name);
    alloc.free(repaired.arguments_json);
    repaired.name = rewritten_name;
    repaired.arguments_json = rewritten_arguments;
    repaired.argument_integrity = .valid;
    return repaired;
}

pub fn isDiscoveryTool(name: []const u8) bool {
    for (discovery_tools) |allowed| {
        if (std.mem.eql(u8, name, allowed)) return true;
    }
    return false;
}

pub fn isLocalInspectionTool(name: []const u8) bool {
    for (local_inspection_tools) |allowed| {
        if (std.mem.eql(u8, name, allowed)) return true;
    }
    return false;
}

pub fn isReadOnlyLocalInspectionTool(name: []const u8) bool {
    for (read_only_local_inspection_tools) |allowed| {
        if (std.mem.eql(u8, name, allowed)) return true;
    }
    return false;
}

pub fn isReadOnlyInspection(prompt: []const u8) bool {
    for (read_only_directives) |directive| {
        if (std.ascii.indexOfIgnoreCase(prompt, directive) != null) return true;
    }
    for (mutating_or_execution_markers) |marker| {
        if (containsWordIgnoreCase(prompt, marker)) return false;
    }
    return true;
}

pub fn isTerminalOnlyRequest(prompt: []const u8) bool {
    var has_execution_marker = false;
    for (terminal_request_markers) |marker| {
        if (containsWordIgnoreCase(prompt, marker)) {
            has_execution_marker = true;
            break;
        }
    }
    if (!has_execution_marker) return false;
    for (negative_execution_directives) |directive| {
        if (std.ascii.indexOfIgnoreCase(prompt, directive) != null) return false;
    }
    for (read_only_directives) |directive| {
        if (std.ascii.indexOfIgnoreCase(prompt, directive) != null) return true;
    }
    for (mutation_markers) |marker| {
        if (containsWordIgnoreCase(prompt, marker)) return false;
    }
    return true;
}

pub fn blocksTerminalOnlyTool(
    provider: model_provider.ProviderId,
    prompt: []const u8,
    tool_messages: []const types.ChatMessage,
    tool_name: []const u8,
) bool {
    return provider == .local and
        isTerminalOnlyRequest(prompt) and
        hasSuccessfulGraphifyRetrieval(tool_messages) and
        !std.mem.eql(u8, tool_name, "terminal");
}

pub fn readOnlyToolMatchesPrompt(prompt: []const u8, name: []const u8) bool {
    if (!isReadOnlyLocalInspectionTool(name)) return false;
    if (std.mem.eql(u8, name, "read_file")) {
        return containsWordIgnoreCase(prompt, "read") and containsFileHint(prompt);
    }
    if (std.mem.eql(u8, name, "list_files")) {
        return containsWordIgnoreCase(prompt, "list") or
            containsWordIgnoreCase(prompt, "directory") or
            containsWordIgnoreCase(prompt, "folder");
    }
    if (std.mem.eql(u8, name, "glob_files")) {
        return containsWordIgnoreCase(prompt, "glob") or
            containsWordIgnoreCase(prompt, "matching paths") or
            containsWordIgnoreCase(prompt, "find files") or
            containsWordIgnoreCase(prompt, "locate files");
    }
    if (std.mem.eql(u8, name, "grep_files")) {
        return containsWordIgnoreCase(prompt, "grep") or
            containsWordIgnoreCase(prompt, "references") or
            containsWordIgnoreCase(prompt, "exact symbol") or
            containsWordIgnoreCase(prompt, "literal");
    }
    if (std.mem.eql(u8, name, "file_info")) {
        return containsWordIgnoreCase(prompt, "exists") or
            containsWordIgnoreCase(prompt, "metadata") or
            containsWordIgnoreCase(prompt, "size") or
            containsWordIgnoreCase(prompt, "modified");
    }
    return containsWordIgnoreCase(prompt, "concept") or
        containsWordIgnoreCase(prompt, "responsibility") or
        containsWordIgnoreCase(prompt, "architecture");
}

fn containsFileHint(prompt: []const u8) bool {
    for ([_][]const u8{ ".md", ".zig", ".ts", ".tsx", ".js", ".jsx", ".py", ".json", ".toml", ".yaml", ".yml", ".txt", ".rs", ".go", ".c", ".h", ".css", ".html", ".sh", ".sql" }) |extension| {
        if (std.ascii.indexOfIgnoreCase(prompt, extension) != null) return true;
    }
    return false;
}

fn extractExplicitCommand(prompt: []const u8) ?[]const u8 {
    const marker = "exactly ";
    const marker_index = std.ascii.indexOfIgnoreCase(prompt, marker) orelse return null;
    const after_marker = prompt[marker_index + marker.len ..];
    const first_backtick = std.mem.indexOfScalar(u8, after_marker, '`') orelse return extractCommandAfterExactly(prompt);
    const after_first = after_marker[first_backtick + 1 ..];
    const second_backtick = std.mem.indexOfScalar(u8, after_first, '`') orelse return extractCommandAfterExactly(prompt);
    const command = std.mem.trim(u8, after_first[0..second_backtick], " \t\r\n");
    return if (command.len > 0) command else null;
}

fn extractCommandAfterExactly(prompt: []const u8) ?[]const u8 {
    const marker = "exactly ";
    const marker_index = std.ascii.indexOfIgnoreCase(prompt, marker) orelse return null;
    var command = prompt[marker_index + marker.len ..];
    for ([_][]const u8{ " and report", " and return", " then", ". do not", " without" }) |delimiter| {
        if (std.ascii.indexOfIgnoreCase(command, delimiter)) |index| command = command[0..index];
    }
    command = std.mem.trim(u8, command, " \t\r\n`\".");
    return if (command.len > 0) command else null;
}

pub fn selectsGraphifyQuery(alloc: Allocator, call: types.ToolCall) bool {
    if (!std.mem.eql(u8, call.name, "mcp_select_tool")) return false;
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, call.arguments_json, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    const selected = parsed.value.object.get("name") orelse return false;
    return selected == .string and std.mem.eql(u8, selected.string, graphify_query_tool);
}

pub fn blocksRedundantGraphifySelection(
    alloc: Allocator,
    provider: model_provider.ProviderId,
    prompt: []const u8,
    tool_messages: []const types.ChatMessage,
    call: types.ToolCall,
) bool {
    return provider == .local and
        requiresGraphify(provider, prompt) and
        hasSuccessfulGraphifyRetrieval(tool_messages) and
        selectsGraphifyQuery(alloc, call);
}

pub fn requiresLocalInspection(provider: model_provider.ProviderId, prompt: []const u8) bool {
    if (provider != .local or !requiresGraphify(provider, prompt)) return false;
    for (task_markers) |marker| {
        if (std.mem.eql(u8, marker, "graphify")) continue;
        if (containsWordIgnoreCase(prompt, marker)) return true;
    }
    return containsFileHint(prompt);
}

pub fn isGraphifyPreflightTool(name: []const u8) bool {
    for (preflight_tools) |allowed| {
        if (std.mem.eql(u8, name, allowed)) return true;
    }
    return false;
}

pub fn blocksBeforeGraphify(
    provider: model_provider.ProviderId,
    prompt: []const u8,
    tool_messages: []const types.ChatMessage,
    tool_name: []const u8,
) bool {
    if (!requiresGraphify(provider, prompt)) return false;
    if (hasSuccessfulGraphifyRetrieval(tool_messages)) return isGraphifyRetrievalTool(tool_name);
    if (isGraphifyRetrievalTool(tool_name) or isDiscoveryTool(tool_name)) return false;
    return true;
}

pub fn blocksFinalResponse(
    provider: model_provider.ProviderId,
    prompt: []const u8,
    tool_messages: []const types.ChatMessage,
) bool {
    if (!requiresGraphify(provider, prompt)) return false;
    if (!hasSuccessfulGraphifyRetrieval(tool_messages)) return true;
    return requiresLocalInspection(provider, prompt) and !hasSuccessfulLocalInspection(tool_messages);
}

pub fn hasSuccessfulGraphifyRetrieval(tool_messages: []const types.ChatMessage) bool {
    for (tool_messages) |message| {
        if (message.role != .tool or message.tool_name == null) continue;
        const status = message.tool_result_status orelse continue;
        if (status == .success and isGraphifyRetrievalTool(message.tool_name.?)) return true;
    }
    return false;
}

pub fn hasSuccessfulTool(tool_messages: []const types.ChatMessage, name: []const u8) bool {
    for (tool_messages) |message| {
        if (message.role != .tool or message.tool_name == null) continue;
        if (message.tool_result_status != .success) continue;
        if (std.mem.eql(u8, message.tool_name.?, name)) return true;
    }
    return false;
}

pub fn hasSuccessfulLocalInspection(tool_messages: []const types.ChatMessage) bool {
    for (tool_messages) |message| {
        if (message.role != .tool or message.tool_name == null) continue;
        if (message.tool_result_status != .success) continue;
        if (isLocalInspectionTool(message.tool_name.?)) return true;
    }
    return false;
}

pub fn appendAutomaticSkillBindings(
    alloc: Allocator,
    provider: model_provider.ProviderId,
    prompt: []const u8,
    skills: []const skill_runtime.Skill,
    bindings: *std.ArrayList(skill_invocation.ExplicitBinding),
) !void {
    if (!requiresGraphify(provider, prompt)) return;

    for ([_][]const u8{ "caveman", "ponytail" }) |wanted| {
        for (skills) |skill| {
            if (!std.mem.eql(u8, skill.name, wanted)) continue;
            var already_added = false;
            for (bindings.items) |binding| {
                if (std.mem.eql(u8, binding.path, skill.path)) {
                    already_added = true;
                    break;
                }
            }
            if (!already_added) {
                try bindings.append(alloc, .{ .name = skill.name, .path = skill.path });
            }
            break;
        }
    }
}

fn containsWordIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or needle.len > haystack.len) return false;
    var offset: usize = 0;
    while (offset <= haystack.len - needle.len) {
        const match_offset = std.ascii.indexOfIgnoreCase(haystack[offset..], needle) orelse return false;
        const index = offset + match_offset;
        const before_is_word = index > 0 and isWordCharacter(haystack[index - 1]);
        const end = index + needle.len;
        const after_is_word = end < haystack.len and isWordCharacter(haystack[end]);
        if (!before_is_word and !after_is_word) return true;
        offset = end;
    }
    return false;
}

fn isWordCharacter(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '_';
}

test "local repository routing gates inspection and loads coding skills" {
    try std.testing.expect(requiresGraphify(.local, "Trace this repository architecture."));
    try std.testing.expect(requiresGraphify(.local, "Read README.md."));
    try std.testing.expect(!requiresGraphify(.local, "What is the weather?"));
    try std.testing.expect(!requiresGraphify(.local, "Show profile settings."));
    try std.testing.expect(!requiresGraphify(.gateway, "Inspect this repository."));
    try std.testing.expect(isGraphifyRetrievalTool(graphify_query_tool));
    try std.testing.expect(isGraphifyRetrievalTool("graphify_query_graph"));
    try std.testing.expect(!isGraphifyRetrievalTool("mcp_graphify_delete_graph"));
    try std.testing.expect(!isGraphifyRetrievalTool("graphify_status"));

    const no_messages = [_]types.ChatMessage{};
    try std.testing.expect(blocksBeforeGraphify(.local, "Fix this code", &no_messages, "read_file"));
    try std.testing.expect(blocksFinalResponse(.local, "Fix this code", &no_messages));
    try std.testing.expect(requiresLocalInspection(.local, "Inspect this repository."));
    try std.testing.expect(!requiresLocalInspection(.local, "Use Graphify to show graph stats."));
    try std.testing.expect(!blocksBeforeGraphify(.local, "Fix this code", &no_messages, "capability_search"));
    try std.testing.expect(!blocksBeforeGraphify(.local, "Fix this code", &no_messages, "graphify_query_graph"));
    try std.testing.expect(isReadOnlyInspection("Read the README and summarize it."));
    try std.testing.expect(isReadOnlyInspection("Read model_provider.zig. Do not edit."));
    try std.testing.expect(!isReadOnlyInspection("Run the focused tests."));
    try std.testing.expect(isTerminalOnlyRequest("Run the focused tests."));
    try std.testing.expect(isTerminalOnlyRequest("Run the focused tests. Do not edit files."));
    try std.testing.expect(!isTerminalOnlyRequest("Do not run exactly `rm -rf /tmp/x`; explain what it would do."));
    try std.testing.expect(!isTerminalOnlyRequest("Do not execute the command; explain it."));
    try std.testing.expect(!isTerminalOnlyRequest("Run tests and fix failures."));
    try std.testing.expect(isReadOnlyLocalInspectionTool("read_file"));
    try std.testing.expect(isReadOnlyLocalInspectionTool("file_info"));
    try std.testing.expect(!isReadOnlyLocalInspectionTool("terminal"));
    try std.testing.expect(isDiscoveryTool("mcp_features"));
    try std.testing.expect(readOnlyToolMatchesPrompt("Read README.md.", "read_file"));
    try std.testing.expect(!readOnlyToolMatchesPrompt("Read README.md.", "glob_files"));
    try std.testing.expect(readOnlyToolMatchesPrompt("Check whether README.md exists.", "file_info"));

    const repaired = try repairGraphifyRoutingCall(
        std.testing.allocator,
        .local,
        "Inspect repository architecture.",
        &no_messages,
        .{ .id = "query", .name = graphify_query_tool, .arguments_json = "{}" },
    );
    defer types.freeToolCall(std.testing.allocator, repaired);
    try std.testing.expectEqualStrings(
        "{\"question\":\"Inspect repository architecture.\"}",
        repaired.arguments_json,
    );

    const selected = try repairGraphifyRoutingCall(
        std.testing.allocator,
        .local,
        "Inspect repository architecture.",
        &no_messages,
        .{ .id = "select", .name = "mcp_select_tool", .arguments_json = "{\"name\":\"graphify\"}" },
    );
    defer types.freeToolCall(std.testing.allocator, selected);
    try std.testing.expectEqualStrings(
        "{\"name\":\"mcp_graphify_query_graph\"}",
        selected.arguments_json,
    );

    const redirected = try repairGraphifyRoutingCall(
        std.testing.allocator,
        .local,
        "Inspect repository architecture.",
        &no_messages,
        .{ .id = "feature", .name = "mcp_features", .arguments_json = "{}" },
    );
    defer types.freeToolCall(std.testing.allocator, redirected);
    try std.testing.expectEqualStrings("mcp_select_tool", redirected.name);
    try std.testing.expectEqualStrings(
        "{\"name\":\"mcp_graphify_query_graph\"}",
        redirected.arguments_json,
    );

    const completed = [_]types.ChatMessage{.{
        .role = .tool,
        .tool_name = "graphify_query_graph",
        .tool_result_status = .success,
    }};
    try std.testing.expect(!blocksBeforeGraphify(.local, "Fix this code", &completed, "read_file"));
    try std.testing.expect(blocksBeforeGraphify(.local, "Fix this code", &completed, graphify_query_tool));
    try std.testing.expect(blocksRedundantGraphifySelection(
        std.testing.allocator,
        .local,
        "Fix this code",
        &completed,
        .{ .id = "select", .name = "mcp_select_tool", .arguments_json = "{\"name\":\"mcp_graphify_query_graph\"}" },
    ));
    try std.testing.expect(blocksTerminalOnlyTool(.local, "Run the focused tests.", &completed, "glob_files"));
    try std.testing.expect(!blocksTerminalOnlyTool(.local, "Run the focused tests.", &completed, "terminal"));
    const repaired_terminal = try repairTerminalExecutionCall(
        std.testing.allocator,
        .local,
        "Run exactly python scripts/build_fx_execution_dataset.py --self-test and report the result.",
        .{ .id = "terminal", .name = "terminal", .arguments_json = "{\"action\":\"exec\",\"command\":\"wrong\",\"timeout_ms\":5000}" },
    );
    defer types.freeToolCall(std.testing.allocator, repaired_terminal);
    try std.testing.expectEqualStrings(
        "{\"action\":\"exec\",\"command\":\"python scripts/build_fx_execution_dataset.py --self-test\",\"timeout_ms\":30000}",
        repaired_terminal.arguments_json,
    );
    try std.testing.expect(blocksFinalResponse(.local, "Fix this code", &completed));
    const inspected = [_]types.ChatMessage{
        completed[0],
        .{ .role = .tool, .tool_name = "read_file", .tool_result_status = .success },
    };
    try std.testing.expect(!blocksFinalResponse(.local, "Fix this code", &inspected));

    const skills = [_]skill_runtime.Skill{
        .{ .name = "caveman", .description = "", .path = "skills/caveman", .source = .global_agents },
        .{ .name = "ponytail", .description = "", .path = "skills/ponytail", .source = .global_agents },
    };
    var bindings: std.ArrayList(skill_invocation.ExplicitBinding) = .empty;
    defer bindings.deinit(std.testing.allocator);
    try appendAutomaticSkillBindings(std.testing.allocator, .local, "Review this code", &skills, &bindings);
    try std.testing.expectEqual(@as(usize, 2), bindings.items.len);
    try std.testing.expectEqualStrings("caveman", bindings.items[0].name);
    try std.testing.expectEqualStrings("ponytail", bindings.items[1].name);
}

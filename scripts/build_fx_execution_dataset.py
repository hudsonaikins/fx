#!/usr/bin/env python3
"""Build a reviewed local-tool execution set from FX traces and seed cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


POST_GRAPHIFY_SYSTEM = (
    "You are a local FX coding agent after runtime Graphify preflight. Call local "
    "tools directly, never use mcp_select_tool for local tools, use workspace-relative "
    "paths, and state only facts supported by tool results."
)
PREFLIGHT_SYSTEM = (
    "You are a local FX coding agent. For repository work, call capability_search "
    "first, select the exact Graphify query tool with mcp_select_tool, call the "
    "selected Graphify tool, then use local tools directly. Never invent tool names "
    "or paths."
)


def python_literal(value: Any) -> str:
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(str(key), ensure_ascii=True) + ":" + python_literal(item)
            for key, item in value.items()
        ) + "}"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def native_call(name: str, arguments: dict[str, Any]) -> str:
    rendered = ",".join(f"{key}={python_literal(value)}" for key, value in arguments.items())
    return f"<|tool_call_start|>[{name}({rendered})]<|tool_call_end|>"


def record(
    example_id: str,
    user: str,
    steps: list[tuple[str, dict[str, Any], str]],
    final: str,
    system: str = POST_GRAPHIFY_SYSTEM,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    tool_names: set[str] = set()
    for index, (name, arguments, result) in enumerate(steps, 1):
        messages.append({"role": "assistant", "content": native_call(name, arguments)})
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"seed_{example_id}_{index}",
                "name": name,
                "content": result,
            }
        )
        tool_names.add(name)
    messages.append({"role": "assistant", "content": final})
    return {
        "messages": messages,
        "metadata": {
            "source": "reviewed_execution_seed",
            "example_id": example_id,
            "tool_count": len(steps),
            "tool_names": sorted(tool_names),
            "format": "lfm2.5-native-tool-calls",
        },
    }


def preflight(
    example_id: str,
    user: str,
    question: str,
    local_steps: list[tuple[str, dict[str, Any], str]],
    final: str,
) -> dict[str, Any]:
    steps = [
        (
            "capability_search",
            {"query": "Graphify repository structure and relevant code paths"},
            '[{"server":"graphify","name":"mcp_graphify_query_graph","description":"Query the repository graph"}]',
        ),
        (
            "mcp_select_tool",
            {"name": "mcp_graphify_query_graph"},
            '{"selected":"mcp_graphify_query_graph","server":"graphify"}',
        ),
        (
            "mcp_graphify_query_graph",
            {"question": question},
            "Relevant graph paths: src/core/config, src/core/agent/runtime, src/gateway.",
        ),
    ]
    return record(example_id, user, steps + local_steps, final, PREFLIGHT_SYSTEM)


def seed_records() -> list[dict[str, Any]]:
    seeds = [
        preflight(
            "seed_preflight_read_provider",
            "Inspect how the local provider is resolved, then report the provider identifier.",
            "How does local provider resolution connect model selection to the OpenAI-compatible gateway?",
            [("read_file", {"path": "src/core/config/model_provider.zig"}, "1 const ProviderId = enum { gateway, codex, grok, local };\n")],
            "The local provider identifier is local, resolved through the model-provider configuration.",
        ),
        preflight(
            "seed_preflight_search_routing",
            "Find local routing rules and name the Graphify gate.",
            "Where does local repository routing decide that Graphify retrieval is required?",
            [("grep_files", {"pattern": "requiresGraphify", "path": "src", "include": "*.zig", "mode": "matches", "head_limit": 20}, "src/core/config/local_routing.zig:120:pub fn requiresGraphify\n")],
            "Local repository routing uses requiresGraphify in src/core/config/local_routing.zig.",
        ),
        preflight(
            "seed_preflight_list_runtime",
            "Inspect runtime files involved in agent execution.",
            "Which runtime modules coordinate tool execution after Graphify retrieval?",
            [("list_files", {"path": "src/core/agent/runtime"}, "assistant_stream.zig\nexecution_memory.zig\norchestrator.zig\n")],
            "The runtime directory contains orchestrator.zig, assistant_stream.zig, and execution_memory.zig.",
        ),
        preflight(
            "seed_preflight_glob_tests",
            "Locate routing tests and report one matching path.",
            "Where are the repository routing evaluation tests?",
            [("glob_files", {"pattern": "tests/evals/*routing*.test.ts", "path": ".", "mode": "matches"}, "tests/evals/github-routing.test.ts\ntests/evals/local-routing.test.ts\n")],
            "Routing evaluations live under tests/evals, including github-routing.test.ts.",
        ),
        preflight(
            "seed_preflight_semantic_tools",
            "Find the tool projection code for the local model.",
            "Which code projects built-in tools into the model request?",
            [("semantic_search", {"query": "model tool projection advertised functions", "path": "src"}, "src/core/tooling/tool_projection.zig\nsrc/core/agent/runtime/orchestrator.zig\n")],
            "Tool projection is implemented in src/core/tooling/tool_projection.zig and consumed by the orchestrator.",
        ),
        preflight(
            "seed_preflight_git_status",
            "Check the current worktree state after Graphify retrieval.",
            "What local runtime files are currently changed?",
            [("terminal", {"request": {"action": "exec", "command": "git status --short", "timeout_ms": 5000}}, " M README.md\n?? scripts/unsloth_fx_train.py\n")],
            "The worktree has README.md modified and the Unsloth training script untracked.",
        ),
        preflight(
            "seed_preflight_read_provider_config",
            "Read the local provider configuration and identify its environment variable.",
            "Which environment variable overrides the local chat endpoint?",
            [("read_file", {"path": "src/gateway/openai_compatible.zig", "start_line": 1, "line_count": 24}, "const chat_url_env = \"FX_LOCAL_CHAT_URL\";\n")],
            "FX_LOCAL_CHAT_URL overrides the local OpenAI-compatible chat endpoint.",
        ),
        preflight(
            "seed_preflight_mcp_config",
            "Inspect MCP configuration without editing it.",
            "How is the Graphify MCP server represented in the local profile?",
            [("read_file", {"path": "src/core/mcp/workspace_config.zig"}, "pub const ServerConfig = struct { name: []const u8, transport: Transport };\n")],
            "The MCP profile represents each server with a name and transport configuration.",
        ),
        record(
            "seed_read_model_provider",
            "Read src/core/config/model_provider.zig and report its provider enum.",
            [("read_file", {"path": "src/core/config/model_provider.zig"}, "const ProviderId = enum { gateway, codex, grok, local };\n")],
            "ProviderId includes gateway, codex, grok, and local.",
        ),
        record(
            "seed_read_local_routing_range",
            "Read the local routing functions around requiresGraphify.",
            [("read_file", {"path": "src/core/config/local_routing.zig", "start_line": 90, "line_count": 45}, "pub fn requiresGraphify(provider: ProviderId, prompt: []const u8) bool { ... }\n")],
            "The routing module checks provider and task markers before requiring Graphify.",
        ),
        record(
            "seed_read_openai_provider",
            "Read the local provider URL constants.",
            [("read_file", {"path": "src/gateway/openai_compatible.zig", "start_line": 8, "line_count": 12}, "const default_chat_url = \"http://127.0.0.1:1234/v1/chat/completions\";\n")],
            "The default local endpoint is LM Studio-compatible loopback URL http://127.0.0.1:1234/v1/chat/completions.",
        ),
        record(
            "seed_read_training_script",
            "Read the LFM target module list in scripts/unsloth_fx_train.py.",
            [("read_file", {"path": "scripts/unsloth_fx_train.py", "start_line": 19, "line_count": 12}, "LFM_TARGET_MODULES = [\"w1\", \"w2\", \"w3\", \"q_proj\", ...]\n")],
            "The training script targets LFM-specific convolution and attention projections.",
        ),
        record(
            "seed_list_core_config",
            "List files under src/core/config.",
            [("list_files", {"path": "src/core/config"}, "local_routing.zig\nmodel_provider.zig\ncontext_limits.zig\n")],
            "src/core/config contains local_routing.zig, model_provider.zig, and context_limits.zig.",
        ),
        record(
            "seed_list_gateway",
            "List gateway implementation files.",
            [("list_files", {"path": "src/gateway"}, "openai_compatible.zig\nclient.zig\nprovider_set.zig\n")],
            "The gateway directory contains openai_compatible.zig, client.zig, and provider_set.zig.",
        ),
        record(
            "seed_list_scripts",
            "List the training-related scripts.",
            [("list_files", {"path": "scripts"}, "export_fx_trajectories.py\nunsloth_fx_train.py\nbuild_fx_execution_dataset.py\n")],
            "Training scripts include export_fx_trajectories.py, unsloth_fx_train.py, and build_fx_execution_dataset.py.",
        ),
        record(
            "seed_glob_zig_routing",
            "Find Zig files named local_routing.",
            [("glob_files", {"pattern": "**/local_routing.zig", "path": ".", "mode": "matches"}, "src/core/config/local_routing.zig\n")],
            "The routing implementation is src/core/config/local_routing.zig.",
        ),
        record(
            "seed_glob_eval_files",
            "Count evaluation files under tests/evals.",
            [("glob_files", {"pattern": "tests/evals/*.test.ts", "path": ".", "mode": "count"}, "count: 31\n")],
            "There are 31 evaluation test files under tests/evals.",
        ),
        record(
            "seed_glob_python_tools",
            "Find Python scripts used for local model work.",
            [("glob_files", {"pattern": "scripts/*lfm*.py", "path": ".", "mode": "matches"}, "scripts/export_fx_trajectories.py\nscripts/unsloth_fx_train.py\n")],
            "The LFM scripts are export_fx_trajectories.py and unsloth_fx_train.py.",
        ),
        record(
            "seed_glob_mcp_sources",
            "Find MCP runtime source files.",
            [("glob_files", {"pattern": "src/core/mcp/*.zig", "path": ".", "mode": "matches"}, "src/core/mcp/mcp_runtime.zig\nsrc/core/mcp/streamable_http.zig\n")],
            "MCP runtime files live under src/core/mcp.",
        ),
        record(
            "seed_grep_local_model",
            "Find references to FX_LOCAL_MODEL in Zig sources.",
            [("grep_files", {"pattern": "FX_LOCAL_MODEL", "path": "src", "include": "*.zig", "mode": "matches", "head_limit": 20}, "src/core/app/app_lifecycle.zig:FX_LOCAL_MODEL\nsrc/acp/server.zig:FX_LOCAL_MODEL\n")],
            "FX_LOCAL_MODEL is read by the application and ACP runtime paths.",
        ),
        record(
            "seed_grep_tool_choice",
            "Find where tool_choice is assigned in the agent runtime.",
            [("grep_files", {"pattern": "tool_choice", "path": "src/core/agent/runtime", "include": "*.zig", "mode": "matches", "head_limit": 30}, "src/core/agent/runtime/orchestrator.zig:3707:const tool_choice\n")],
            "The orchestrator assigns tool_choice while preparing each model request.",
        ),
        record(
            "seed_grep_graphify_query",
            "Find the exact Graphify query tool name.",
            [("grep_files", {"pattern": "mcp_graphify_query_graph", "path": "src", "include": "*.zig", "mode": "files_with_matches", "head_limit": 20}, "src/core/config/local_routing.zig\nsrc/core/agent/runtime/orchestrator.zig\n")],
            "The exact Graphify query tool name is mcp_graphify_query_graph.",
        ),
        record(
            "seed_grep_native_tags",
            "Find the native Liquid tool-call delimiters.",
            [("grep_files", {"pattern": "tool_call_start", "path": "src/gateway", "include": "*.zig", "mode": "matches", "head_limit": 20}, "src/gateway/openai_compatible.zig:const fallback_tool_tags\n")],
            "The local provider recognizes Liquid native tool calls between tool_call_start and tool_call_end tags.",
        ),
        record(
            "seed_semantic_routing",
            "Find code responsible for local repository routing.",
            [("semantic_search", {"query": "local provider repository Graphify routing", "path": "src"}, "src/core/config/local_routing.zig\nsrc/core/agent/runtime/orchestrator.zig\n")],
            "Local repository routing is split between local_routing.zig and the agent orchestrator.",
        ),
        record(
            "seed_semantic_training",
            "Find the local fine-tuning entry point.",
            [("semantic_search", {"query": "Unsloth QLoRA LFM trajectory training", "path": "scripts"}, "scripts/unsloth_fx_train.py\nscripts/export_fx_trajectories.py\n")],
            "unsloth_fx_train.py trains the adapter and export_fx_trajectories.py prepares traces.",
        ),
        record(
            "seed_semantic_mcp",
            "Find MCP tool discovery and selection code.",
            [("semantic_search", {"query": "MCP capability discovery exact tool selection", "path": "src/core"}, "src/core/tooling/tool_mcp_dispatch.zig\nsrc/builtins/tools.zig\n")],
            "MCP discovery and selection are implemented in tool_mcp_dispatch.zig and described in builtins/tools.zig.",
        ),
        record(
            "seed_file_info_graph",
            "Check whether graphify-out/graph.json exists.",
            [("file_info", {"path": "graphify-out/graph.json"}, "type: file\nsize: 27699535 bytes\n")],
            "graphify-out/graph.json exists as a regular file.",
        ),
        record(
            "seed_file_info_script",
            "Inspect metadata for scripts/unsloth_fx_train.py.",
            [("file_info", {"path": "scripts/unsloth_fx_train.py"}, "type: file\nsize: 9480 bytes\n")],
            "scripts/unsloth_fx_train.py is a regular file.",
        ),
        record(
            "seed_terminal_status",
            "Run git status --short and summarize the worktree.",
            [("terminal", {"request": {"action": "exec", "command": "git status --short", "timeout_ms": 5000}}, " M README.md\n?? scripts/unsloth_fx_train.py\n")],
            "The worktree has a modified README.md and an untracked Unsloth training script.",
        ),
        record(
            "seed_terminal_focused_test",
            "Run the exporter self-test.",
            [("terminal", {"request": {"action": "exec", "command": "python scripts/export_fx_trajectories.py --self-test", "timeout_ms": 30000}}, "self-test=passed\n")],
            "The trajectory exporter self-test passed.",
        ),
        record(
            "seed_terminal_zig_build",
            "Build the current Zig checkout.",
            [("terminal", {"request": {"action": "exec", "command": "zig build", "timeout_ms": 120000}}, "build completed successfully\n")],
            "The Zig build completed successfully.",
        ),
        record(
            "seed_terminal_git_log",
            "Show the latest commit subject.",
            [("terminal", {"request": {"action": "exec", "command": "git log -1 --format=%s", "timeout_ms": 5000}}, "Add local model routing\n")],
            "The latest commit subject is Add local model routing.",
        ),
        record(
            "seed_terminal_exact_exporter",
            "Run exactly `python scripts/export_fx_trajectories.py --self-test` and report the result.",
            [("terminal", {"request": {"action": "exec", "command": "python scripts/export_fx_trajectories.py --self-test", "timeout_ms": 30000}}, "self-test=passed\n")],
            "The exporter self-test passed.",
        ),
        record(
            "seed_terminal_exact_dataset",
            "Execute this exact command: python scripts/build_fx_execution_dataset.py --self-test",
            [("terminal", {"request": {"action": "exec", "command": "python scripts/build_fx_execution_dataset.py --self-test", "timeout_ms": 30000}}, "self-test=passed seeds=36\n")],
            "The execution dataset self-test passed with 36 seed records.",
        ),
        record(
            "seed_terminal_exact_compile",
            "Run `python -m py_compile scripts/export_fx_trajectories.py scripts/build_fx_execution_dataset.py`.",
            [("terminal", {"request": {"action": "exec", "command": "python -m py_compile scripts/export_fx_trajectories.py scripts/build_fx_execution_dataset.py", "timeout_ms": 30000}}, "")],
            "Python compilation completed successfully.",
        ),
        record(
            "seed_terminal_exact_diff_check",
            "Run exactly `git diff --check` and report whether it passed.",
            [("terminal", {"request": {"action": "exec", "command": "git diff --check", "timeout_ms": 30000}}, "")],
            "git diff --check passed.",
        ),
        record(
            "seed_terminal_exact_zig_format",
            "Run this command without changing files: zig fmt --check src/core/config/local_routing.zig",
            [("terminal", {"request": {"action": "exec", "command": "zig fmt --check src/core/config/local_routing.zig", "timeout_ms": 30000}}, "")],
            "Zig formatting check passed.",
        ),
        record(
            "seed_terminal_exact_py_version",
            "Run exactly `python --version` and report the output.",
            [("terminal", {"request": {"action": "exec", "command": "python --version", "timeout_ms": 5000}}, "Python 3.x\n")],
            "Python version command completed.",
        ),
        record(
            "seed_terminal_exact_status",
            "Execute exactly `git status --short`.",
            [("terminal", {"request": {"action": "exec", "command": "git status --short", "timeout_ms": 5000}}, " M README.md\n")],
            "git status --short completed.",
        ),
        record(
            "seed_terminal_exact_zig_build",
            "Build the checkout by running exactly `zig build`.",
            [("terminal", {"request": {"action": "exec", "command": "zig build", "timeout_ms": 120000}}, "build completed successfully\n")],
            "The Zig build completed successfully.",
        ),
        record(
            "seed_edit_readme",
            "Change one exact local endpoint example in README.md.",
            [("edit_file", {"path": "README.md", "old_string": "FX_LOCAL_MODEL=lfm-q4-128k", "new_string": "FX_LOCAL_MODEL=lfm-stock-q8"}, "edited: README.md\n")],
            "README.md was edited by replacing the exact local model example.",
        ),
        record(
            "seed_edit_config",
            "Replace one exact default model identifier in the local config.",
            [("edit_file", {"path": "src/core/config/model_provider.zig", "old_string": "lfm-q4-128k", "new_string": "lfm-stock-q8"}, "edited: src/core/config/model_provider.zig\n")],
            "The exact model identifier was replaced in the local provider configuration.",
        ),
        record(
            "seed_write_eval_note",
            "Create a small local evaluation note at tmp/fx-eval.txt.",
            [("write_file", {"path": "tmp/fx-eval.txt", "content": "tool=read_file\nstatus=pass\n"}, "wrote: tmp/fx-eval.txt\n")],
            "Created tmp/fx-eval.txt with the local evaluation result.",
        ),
        record(
            "seed_write_config_fixture",
            "Create a test fixture at tests/fixtures/local-model.json.",
            [("write_file", {"path": "tests/fixtures/local-model.json", "content": "{\n  \"model\": \"lfm-stock-q8\",\n  \"context\": 128000\n}\n"}, "wrote: tests/fixtures/local-model.json\n")],
            "Created the local-model.json fixture with the Q8 model and 128000 context.",
        ),
    ]
    for item in seeds:
        digest = hashlib.sha256(json.dumps(item["messages"], sort_keys=True).encode()).hexdigest()[:16]
        item["metadata"]["example_id"] = item["metadata"].get("example_id", digest)
    return seeds


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
                if isinstance(value, dict) and isinstance(value.get("messages"), list):
                    records.append(value)
    return records


def write_dataset(output: Path, records: list[dict[str, Any]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in records:
            stream.write(json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=False, default=Path("fx-execution-training.jsonl"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    seeds = seed_records()
    if args.self_test:
        assert len(seeds) >= 35
        assert {"read_file", "grep_files", "glob_files", "list_files", "semantic_search", "terminal"} <= {
            name for item in seeds for name in item["metadata"]["tool_names"]
        }
        tool_messages = [
            message
            for item in seeds
            for message in item["messages"]
            if message["role"] == "assistant" and message["content"].startswith("<|tool_call_start|>")
        ]
        assert len(tool_messages) == sum(item["metadata"]["tool_count"] for item in seeds)
        print(f"self-test=passed seeds={len(seeds)}")
        return 0

    combined = seeds + load_records(args.input)
    unique: dict[str, dict[str, Any]] = {}
    for item in combined:
        key = hashlib.sha256(json.dumps(item.get("messages", []), sort_keys=True).encode()).hexdigest()
        unique.setdefault(key, item)
    records = list(unique.values())
    write_dataset(args.output, records)
    print(f"seeds={len(seeds)} imported={len(combined) - len(seeds)} records={len(records)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

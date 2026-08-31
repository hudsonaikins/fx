#!/usr/bin/env python3
"""Export healthy local FX turns as redacted Liquid-native SFT records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


NATIVE_SYSTEM = (
    "You are a local FX coding agent after runtime Graphify preflight. Call local "
    "tools directly, never use mcp_select_tool for local tools, use workspace-relative "
    "paths, and state only facts supported by tool results."
)
MAX_TOOL_OUTPUT_CHARS = 16_000
MAX_ARGUMENT_CHARS = 64_000
ERROR_MARKERS = (
    "tool_review_held",
    "tool_execution_failed",
    "review_unavailable",
    "graph.json not found",
    "Error executing query_graph",
)
SECRET_KEY_PATTERN = (
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"password|passwd|secret(?:[_-]?key)?|client[_-]?secret|"
    r"authorization|proxy[_-]?authorization|cookie|set[_-]?cookie|"
    r"github[_-]?token|hf[_-]?token|token|credential|credentials)"
)
SECRET_RE = re.compile(
    r"(?ix)"
    r"(?:"
    r"\b(?:authorization|proxy[_-]?authorization)\b\s*[\"']?\s*:\s*[\"']?(?:bearer|token)\s+"
    r"|\bbearer\s+"
    r"|\b" + SECRET_KEY_PATTERN + r"\b\s*[\"']?\s*[:=]\s*[\"']?"
    r")[^\s,;}\"']+"
)
TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|(?:ghp|gho|ghs|ghr)[_-][A-Za-z0-9_-]+|"
    r"github_pat[_-][A-Za-z0-9_-]+|xox[baprs]-[A-Za-z0-9_-]+|"
    r"hf_[A-Za-z0-9_-]+)\b"
)
SENSITIVE_KEY_NAMES = frozenset(
    {
        "apikey",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "password",
        "passwd",
        "secret",
        "secretkey",
        "clientsecret",
        "authorization",
        "proxyauthorization",
        "cookie",
        "setcookie",
        "token",
        "credential",
        "credentials",
        "githubtoken",
        "hftoken",
    }
)
USER_DIRECTORY = "Users"
USER_ROOT_RE = re.compile(
    r"(?i)(?:/mnt/[a-z]/" + USER_DIRECTORY + r"/[^/\\]+|[A-Za-z]:[\\/]"
    + USER_DIRECTORY
    + r"[\\/]+[^/\\]+)"
)
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def path_variants(path: str) -> list[str]:
    variants = [path]
    if path.startswith("/mnt/") and len(path) > 6 and path[5].isalpha() and path[6] == "/":
        variants.append(path[5].upper() + ":" + path[6:].replace("/", "\\"))
    if len(path) > 2 and path[1] == ":" and path[2] in "/\\":
        variants.append("/mnt/" + path[0].lower() + "/" + path[3:].replace("\\", "/"))
    if "/" in path:
        variants.append(path.replace("/", "\\"))
    if "\\" in path:
        variants.append(path.replace("\\", "/"))
    return list(dict.fromkeys(variants))


def redact_text(value: str, roots: list[str]) -> str:
    output = value
    for root in roots:
        if not root:
            continue
        for variant in path_variants(root.rstrip("/\\")):
            output = output.replace(variant, "<WORKSPACE>")
    output = USER_ROOT_RE.sub("<USER>", output)
    output = SECRET_RE.sub("<REDACTED>", output)
    return TOKEN_RE.sub("<REDACTED>", output)


def is_sensitive_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in SENSITIVE_KEY_NAMES


def redact_value(value: Any, roots: list[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, roots)
    if isinstance(value, list):
        return [redact_value(item, roots) for item in value]
    if isinstance(value, dict):
        return {
            key: "<REDACTED>" if is_sensitive_key(key) else redact_value(item, roots)
            for key, item in value.items()
        }
    return value


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
        pairs = []
        for key, item in value.items():
            pairs.append(
                json.dumps(str(key), ensure_ascii=True)
                + ":"
                + python_literal(item)
            )
        return "{" + ",".join(pairs) + "}"
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def native_tool_call_content(calls: list[dict[str, Any]], roots: list[str]) -> str | None:
    rendered: list[str] = []
    for call in calls:
        name = call.get("name")
        if not isinstance(name, str) or not TOOL_NAME_RE.fullmatch(name):
            return None
        raw_arguments = call.get("arguments_json") or "{}"
        if not isinstance(raw_arguments, str) or len(raw_arguments) > MAX_ARGUMENT_CHARS:
            return None
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return None
        if not isinstance(arguments, dict):
            return None
        arguments = redact_value(arguments, roots)
        rendered_pairs = []
        for key, value in arguments.items():
            if not isinstance(key, str) or not TOOL_NAME_RE.fullmatch(key):
                return None
            rendered_pairs.append(key + "=" + python_literal(value))
        rendered_arguments = ",".join(rendered_pairs)
        rendered.append(f"{name}({rendered_arguments})")
    if not rendered:
        return None
    return "<|tool_call_start|>[" + ",".join(rendered) + "]<|tool_call_end|>"


def healthy_result(result: dict[str, Any]) -> bool:
    if result.get("status") != "success":
        return False
    output = result.get("output")
    if not isinstance(output, str):
        return False
    return not any(marker in output for marker in ERROR_MARKERS)


def is_runtime_preflight_call(call: dict[str, Any]) -> bool:
    call_id = call.get("id")
    return isinstance(call_id, str) and call_id.startswith("fx_local_graphify_")


def bounded_output(value: str, roots: list[str]) -> str:
    output = redact_text(value, roots)
    if len(output) <= MAX_TOOL_OUTPUT_CHARS:
        return output
    return output[:MAX_TOOL_OUTPUT_CHARS] + "\n<TRUNCATED>"


def build_record(
    session_id: str,
    metadata: dict[str, Any],
    turn: dict[str, Any],
    explicit_redact_root: str | None = None,
) -> dict[str, Any] | None:
    if turn.get("kind") != "assistant":
        return None
    user = turn.get("user")
    execution = turn.get("execution")
    final_text = turn.get("assistant")
    if not isinstance(user, dict) or not isinstance(execution, dict) or not isinstance(final_text, str):
        return None
    if not final_text.strip():
        return None

    workspace_root = explicit_redact_root or metadata.get("workspace_root") or ""
    roots = [workspace_root, os.environ.get("HOME", "")]
    messages: list[dict[str, Any]] = [{"role": "system", "content": NATIVE_SYSTEM}]
    user_text = user.get("text")
    if not isinstance(user_text, str) or not user_text.strip():
        return None
    messages.append({"role": "user", "content": redact_text(user_text, roots)})

    tool_names: set[str] = set()
    tool_count = 0
    for step in execution.get("tool_steps", []):
        if not isinstance(step, dict):
            return None
        calls = step.get("tool_calls") or []
        results = step.get("tool_results") or []
        if not calls:
            continue
        if not isinstance(calls, list) or not isinstance(results, list):
            return None
        if calls and all(isinstance(call, dict) and is_runtime_preflight_call(call) for call in calls):
            continue
        if any(isinstance(call, dict) and is_runtime_preflight_call(call) for call in calls):
            return None
        content = native_tool_call_content(calls, roots)
        if content is None:
            return None
        assistant_text = step.get("assistant")
        if isinstance(assistant_text, str):
            content += redact_text(assistant_text, roots)
        messages.append({"role": "assistant", "content": content})
        for result in results:
            if not isinstance(result, dict) or not healthy_result(result):
                return None
            name = result.get("tool_name")
            output = result.get("output")
            if not isinstance(name, str) or not isinstance(output, str):
                return None
            tool_names.add(name)
            tool_count += 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.get("tool_call_id", ""),
                    "name": name,
                    "content": bounded_output(output, roots),
                }
            )

    if tool_count == 0:
        return None
    messages.append({"role": "assistant", "content": redact_text(final_text, roots)})
    model = metadata.get("preferences", {}).get("model")
    provider = metadata.get("preferences", {}).get("provider")
    record = {
        "messages": messages,
        "metadata": {
            "source_session": session_id,
            "provider": provider,
            "model": model,
            "tool_count": tool_count,
            "tool_names": sorted(tool_names),
            "format": "lfm2.5-native-tool-calls",
            "redaction_version": 2,
        },
    }
    digest = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]
    record["metadata"]["example_id"] = digest
    return record


def load_session_turns(session_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    try:
        metadata = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    turns: list[dict[str, Any]] = []
    try:
        with (session_dir / "events.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("kind") == "history_turn_committed":
                    candidate = event.get("payload", {}).get("turn")
                    if isinstance(candidate, dict):
                        turns.append(candidate)
    except OSError:
        return None
    if not turns:
        return None
    return metadata, turns


def export(args: argparse.Namespace) -> int:
    sessions_dir = Path(args.sessions_dir).expanduser()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    skipped = 0
    for session_dir in sorted(sessions_dir.iterdir() if sessions_dir.is_dir() else []):
        if not session_dir.is_dir() or session_dir.name in {"latest", "index.pending"}:
            continue
        loaded = load_session_turns(session_dir)
        if loaded is None:
            skipped += 1
            continue
        metadata, turns = loaded
        preferences = metadata.get("preferences", {})
        if args.provider and preferences.get("provider") != args.provider:
            continue
        if args.model_prefix and not str(preferences.get("model", "")).startswith(args.model_prefix):
            continue
        for turn in turns:
            record = build_record(session_dir.name, metadata, turn, args.redact_root)
            if record is None:
                skipped += 1
                continue
            example_id = record["metadata"]["example_id"]
            if example_id not in seen:
                seen.add(example_id)
                records.append(record)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
    print(f"exported={len(records)} skipped={skipped} output={output_path}")
    return 0


def self_test() -> int:
    metadata = {
        "workspace_root": "/workspace",
        "preferences": {"provider": "local", "model": "lfm-q8-128k"},
    }
    turn = {
        "kind": "assistant",
        "user": {"text": "Inspect /workspace/src."},
        "assistant": "Done.",
        "execution": {
            "tool_steps": [
                {
                    "tool_calls": [
                        {
                            "id": "fx_local_graphify_query",
                            "name": "mcp_graphify_query_graph",
                            "arguments_json": '{"question":"inspect"}',
                        }
                    ],
                    "tool_results": [
                        {
                            "tool_call_id": "fx_local_graphify_query",
                            "tool_name": "mcp_graphify_query_graph",
                            "status": "success",
                            "output": "graph result",
                        }
                    ],
                },
                {
                    "tool_calls": [
                        {
                            "name": "read_file",
                            "arguments_json": '{"path":"/workspace/src/main.zig"}',
                        }
                    ],
                    "tool_results": [
                        {
                            "tool_call_id": "call_1",
                            "tool_name": "read_file",
                            "status": "success",
                            "output": "source",
                        }
                    ],
                }
            ]
        },
    }
    record = build_record("test", metadata, turn)
    assert record is not None
    assert record["metadata"]["tool_names"] == ["read_file"]
    assert "mcp_graphify_query_graph" not in str(record["messages"])
    assert "<WORKSPACE>" in record["messages"][1]["content"]
    assert "<|tool_call_start|>[read_file(path=\"<WORKSPACE>/src/main.zig\")]<|tool_call_end|>" in record["messages"][2]["content"]
    wsl_home = "/mnt/c/" + USER_DIRECTORY + "/tester/.codex"
    windows_home = "C:\\" + USER_DIRECTORY + "\\tester"
    assert redact_text(wsl_home, [windows_home]) == "<WORKSPACE>/.codex"
    redacted = redact_text(
        "Authorization: token gho_abcdefghijklmnopqrstuvwxyz "
        "github_pat_11AAAA_secret token=plain-secret api_key=plain-api-key",
        [],
    )
    assert "gho_" not in redacted
    assert "github_pat_" not in redacted
    assert "plain-secret" not in redacted
    assert "plain-api-key" not in redacted
    structured = redact_value(
        {"token": "structured-secret", "token_count": 128, "authorization_docs": "public"},
        [],
    )
    assert structured["token"] == "<REDACTED>"
    assert structured["token_count"] == 128
    assert structured["authorization_docs"] == "public"
    turn["execution"]["tool_steps"][1]["tool_results"][0]["status"] = "failure"
    assert build_record("test", metadata, turn) is None
    print("self-test=passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-dir", default="~/.fx/sessions")
    parser.add_argument("--output", default="fx-trajectories.jsonl")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--redact-root", default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    return export(args)


if __name__ == "__main__":
    sys.exit(main())

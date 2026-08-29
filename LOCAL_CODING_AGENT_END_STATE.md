# FX Local Coding Agent End State

Status: target contract
Owner: FX local-model work
Date: 2026-08-29

## Current Checkpoint

- Corpus: 100 records, combining 44 curated execution examples with 56 exported healthy FX sessions.
- Candidate: one local Unsloth QLoRA epoch, exported as Q8 GGUF, trained on an 80-record split with 20 records held out by the trainer.
- A/B snapshot: v1 completed 7 of 8 probes; the new candidate completed 2 of 8. Candidate rejected and left inactive.
- Active rollback model: `lfm-fx-execution-v1` at 128,000 context.
- Permission gate: local `--auto` review now routes through LM Studio and returned `review_caution` for a held action; PTY interactive approval and denial both verified. The smoke ended at the agent-step limit after denied follow-up variants, so process-completion quality remains unproven.
- Decision: stop model tuning here. Keep v1 active; finish focused real-task evaluation, CI, and pilot gates before promotion.

## Purpose

Prevent endless local-model development. This document defines the finish line for the first FX local coding-agent release. Work stops when these gates pass. Later work requires a measured failure, security issue, or explicit product change.

## Product Position

Target result: evaluation-backed local coding-agent pilot for private internal dogfooding.

This is not a replacement for frontier hosted coding agents. It should handle repository search, file inspection, Graphify-backed context, bounded terminal commands, and targeted edits. It should not claim reliable autonomous work across large ambiguous refactors.

## Target Architecture

- Local model: Liquid LFM2.5-8B-A1B, deployed as the Q8 GGUF artifact through LM Studio.
- Hardware target: RTX 5070 Ti with 16 GB VRAM.
- Context ceiling: 128,000 tokens in LM Studio. Large context remains a ceiling, not permission to inject an entire repository into every prompt.
- Training: local Unsloth QLoRA adapter training, then Q8 GGUF export for deployment. Do not train Q8 weights directly.
- Retrieval: Graphify MCP preflight before repository inspection or execution.
- Skills: Caveman and Ponytail loaded for local coding requests.
- Tool flow: Graphify preflight, narrow local tool projection, exact tool arguments, permissioned execution, result verification.
- Runtime: FX local provider through the OpenAI-compatible LM Studio endpoint.
- Baseline: `lfm-fx-execution-v1` remains rollback target. Experimental candidates stay inactive until they pass A/B gates.

## Definition Of Complete

### Runtime

- LM Studio loads the selected Q8 model at 128,000 context.
- FX uses the freshly built `./zig-out/bin/fx` binary.
- Graphify retrieval succeeds before local repository tools.
- Read, list, grep, file metadata, terminal, and targeted edit flows execute through native provider tool calls.
- Explicit commands preserve user command text. No model-invented replacement command executes.
- Normal execution uses FX permission policy. `--yolo` is smoke-test-only.

### Quality Gates

Promote a fine-tuned candidate only when it meets every gate below on a held-out real-task evaluation set:

- Read-only tasks: 100% no-write behavior.
- Graphify preflight compliance: at least 95%.
- Valid tool-call schema and argument parsing: at least 95%.
- Exact terminal-command execution: at least 90%.
- Targeted edit plus verification success: at least 80%.
- Candidate does not regress baseline read, search, safety, or completion success.
- Permission review works in the intended interactive workflow.
- Full CI passes for the exact feature commit on required native runners.

Measure task success, tool selection, argument exactness, unauthorized mutations, retries, time to first token, generation speed, and p95 end-to-end latency. Do not use model loss alone as a promotion decision.

## Training Boundary

Collect 100 to 300 healthy FX trajectories containing Graphify retrieval, local inspection, terminal execution, edits, failures, and recovery. Keep training and held-out evaluation records separate.

Run one local Unsloth QLoRA pass first. Export Q8 GGUF. Compare candidate against `lfm-fx-execution-v1`. Stop tuning after candidate passes gates. More epochs, larger adapters, DPO, GRPO, or model replacement require evidence from a failed gate.

## Explicit Non-Goals

- Cloud fine-tuning.
- Full-model pretraining.
- SSD-backed KV-cache engineering for this release.
- VLM support.
- DFlash or speculative decoding integration before baseline task quality is proven.
- Automatic destructive execution.
- Entire-repository prompt injection as a substitute for Graphify retrieval.
- Multi-agent orchestration or a second execution path.

## Rollback

Rollback to `lfm-fx-execution-v1` if any candidate regresses read accuracy, tool routing, command exactness, safety, or latency. Keep rejected candidates installed for comparison, but never active by default.

If local model quality remains below gates, retain the architecture as a retrieval and tool-contract experiment. Do not keep adding routing heuristics to compensate for a model that cannot meet the task threshold.

## Finish Sequence

1. Expand healthy trajectory corpus to 100 to 300 records.
2. Train one local QLoRA adapter with Unsloth.
3. Export Q8 GGUF and load at 128K in LM Studio.
4. Run held-out A/B evaluation against v1.
5. Fix interactive permission review and rerun execution tests. Complete: local automatic reviewer plus PTY approval/denial smoke verified.
6. Run focused checks, build the binary, exercise real FX flows, and wait for Full CI on the exact commit.
7. Declare internal pilot complete and freeze architecture.

After step 7, only bug fixes, security fixes, dependency updates, and changes justified by a failed metric reopen development.

## Research Basis

- [LFM2.5 model card](https://huggingface.co/unsloth/LFM2.5-8B-A1B): 8.3B total parameters, 1.5B active parameters, 131,072-token context, native tool-call format, and recommendation to use retrieval for heavy programming work.
- [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization): quantized bases are normally adapted with PEFT; QLoRA uses a 4-bit base with trainable LoRA adapters.
- [Unsloth LM Studio deployment](https://unsloth.ai/docs/basics/inference-and-deployment/lm-studio): export the fine-tuned model to GGUF for local LM Studio deployment.
- [FX MCP documentation](https://fx.sh/docs/capabilities/mcp): MCP provides the capability discovery and tool integration boundary used by FX.

# Terminal-Bench AgentBeats Integration Detail

## Overview

In this repository, we have fully integrated Terminal-Bench with AgentBeats platform:

- **Green Agent**: Fully AgentBeats-compatible with `tools.py` wrapper
- **White Agent**: Fully AgentBeats-compatible with `tools.py` wrapper

Both agents now use AgentBeats' standard `@ab.tool` pattern, enabling full integration with the AgentBeats platform including `load_scenario` and `run_scenario` commands.

## Architecture

```
AgentBeats Backend
       ↓ battle_start
Green Agent (tools.py) → TerminalBenchGreenAgentExecutor → Terminal-Bench Harness
       ↓ A2A message
White Agent (tools.py) → solve_terminal_bench_task → MCP Server (task-scoped)
       ↓ bash commands
Docker Container (per task)
```

## Integration Details

### 1. Green Agent (`agents/green_agent/tools.py`)

The green agent wraps the original `TerminalBenchGreenAgentExecutor` in an `@ab.tool`:

- **Entrypoint**: `start_terminal_bench_battle(battle_start_json: str)`
- **Functionality**:
  - Parses `battle_start` JSON from AgentBeats backend
  - Extracts participant agent URL and task configuration
  - Runs Terminal-Bench harness
  - Reports detailed results back to backend (including formatted message)
- **Key Feature**: Maintains code-driven approach - all harness logic preserved

### 2. White Agent (`agents/white_agent/tools.py`)

The white agent is now fully integrated with AgentBeats:

- **Tool**: `solve_terminal_bench_task(task_message: str)`
- **Functionality**:
  - Extracts MCP server URL from Terminal-Bench plain text messages
  - Connects dynamically to task-scoped MCP servers
  - Uses existing `solve_task_with_llm_and_mcp` logic
  - Returns task completion results
- **Key Feature**: Wraps existing white agent logic in `@ab.tool` for AgentBeats compatibility

The agent card description guides the LLM to automatically use this tool when receiving Terminal-Bench messages.

## Configuration

**scenario.toml**:
```toml
# Green agent - AgentBeats compatible
[[agents]]
name = "Green Agent"
card = "agents/green_agent/agent_card.toml"
tools = ["agents/green_agent/tools.py"]
is_green = true

# White agent - Fully AgentBeats compatible
[[agents]]
name = "White Agent"
card = "agents/white_agent/agent_card.toml"
tools = ["agents/white_agent/tools.py"]  # Uses solve_terminal_bench_task tool
model_type = "openai"
model_name = "gpt-4o-mini"
```

**config.toml** (Terminal-Bench specific settings):
The green agent loads configuration from `config.toml` including:
- `evaluation.task_ids`: List of tasks to evaluate (default: `["hello-world","create-bucket"]`)
- `evaluation.n_attempts`: Number of attempts per task (default: 1)
- `evaluation.n_concurrent_trials`: Concurrent trial limit (default: 1)
- `evaluation.timeout_multiplier`: Timeout multiplier (default: 1.0)
- `dataset.name`: Dataset name (default: "terminal-bench-core")
- `dataset.version`: Dataset version (default: "0.1.1")

To customize which tasks to run, edit the `evaluation.task_ids` section in `config.toml` in the `scenarios/terminal_bench/` directory.

**Note**: The `config.toml` file is heavily commented with usage indicators showing which settings apply to standalone mode, AgentBeats mode, or both. Settings like `green_agent.*` and `white_agent.*` are only used in standalone mode - AgentBeats uses `scenario.toml` for those.

## Backend Reporting

The green agent reports comprehensive results:

```python
{
    "is_result": True,
    "message": "Terminal-Bench evaluation completed",
    "timestamp": "2025-01-XX...",
    "reported_by": "Terminal-Bench Green Agent",
    "detail": {
        "accuracy": 0.67,  # Overall accuracy
        "n_resolved": 8,
        "n_unresolved": 4,
        "task_config": {...},
        "participant_url": "http://..."
    },
    "markdown_content": "Terminal-Bench Evaluation Results\n..."
}
```

The `markdown_content` field contains the formatted results summary and is rendered as rich markdown in the AgentBeats frontend, preserving line breaks and formatting. It includes:
- Overall weighted score
- Scores by difficulty (Easy/Medium/Hard)
- Per-task breakdown with test results
- Token usage per task
- Failure mode analysis (when applicable)

## Run Evaluation

For detailed instructions on running Terminal-Bench evaluations with AgentBeats, see the main [README.md](../../README.md) in the root directory. The README includes:
- Step-by-step setup instructions
- Configuration guide for API keys
- Platform deployment instructions
- How to create and run battles using the web UI

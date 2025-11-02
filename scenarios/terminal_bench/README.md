# Terminal-Bench Green Agent

A green agent that evaluates other agents on [terminal-bench](https://www.tbench.ai/) using the A2A protocol.

**Note**: This README documents the standalone Terminal-Bench evaluation mode. For AgentBeats Platform integration, see [INTEGRATION.md](INTEGRATION.md).

## Overview

This project implements a **green agent** (evaluator) that runs terminal-bench to evaluate **white agents** (agents under test). Communication between agents uses the A2A protocol, and white agents execute bash commands via task-scoped MCP servers.

**Evaluation Modes**:
- **Standalone Mode** (this README): Run evaluations directly without AgentBeats platform
- **AgentBeats Platform Mode** (see [INTEGRATION.md](INTEGRATION.md)): Run evaluations through AgentBeats with web UI

### Key Features

- **Multi-agent evaluation framework** using A2A protocol
- **Task-scoped MCP servers** for isolated bash command execution
- **Docker-based sandboxing** for secure task environments
- **Flexible configuration** via TOML
- **Comprehensive logging** of agent interactions and command execution

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐         ┌─────────────┐
│   Kickoff   │  A2A    │ Green Agent  │  A2A    │ A2A Adapter │   MCP   │ White Agent │
│  (kickoff)  ├────────►│ (evaluator)  ├────────►│  (bridge)   ├────────►│  (solver)   │
└─────────────┘         └──────────────┘         └─────────────┘         └─────────────┘
                              │                          │
                              │                          │
                              ▼                          ▼
                        Terminal-Bench            Task MCP Server
                          Harness                 (bash execution)
```

### Components

#### 1. Kickoff (`src/kickoff.py`)

- Entry point that initiates evaluation
- Sends evaluation configuration to green agent via A2A
- Validates both agents are running before starting

#### 2. Green Agent (`src/green_agent/`)

- **`green_agent.py`**: Evaluator that orchestrates terminal-bench harness
- **`task_mcp_server.py`**: Creates and manages task-scoped MCP servers
- Receives evaluation requests via A2A
- Manages task lifecycle and collects results

#### 3. A2A Adapter (`src/adapters/a2a_adapter.py`)

- Bridge between terminal-bench harness and white agent
- Implements terminal-bench's agent interface
- Translates terminal-bench calls to A2A messages
- Creates unique MCP server for each task

#### 4. White Agent (`white_agent/`)

- **`white_agent.py`**: Example agent implementation with A2A interface
- **`white_agent_helpers.py`**: MCP connection utilities and LLM integration
- Receives tasks via A2A
- Connects to task-specific MCP server
- Uses LLM to solve tasks by executing bash commands

### Evaluation Flow

1. **Initialization**

   - Kickoff validates both agents are accessible
   - Configuration loaded from `config.toml`

2. **Task Assignment**

   - Kickoff sends evaluation request to green agent
   - Green agent starts terminal-bench harness
   - For each task, harness creates Docker container

3. **Task Execution**

   - A2A adapter creates unique MCP server for the task
   - Adapter sends task instruction + MCP URL to white agent via A2A
   - White agent connects to MCP server
   - White agent iteratively executes bash commands via MCP
   - MCP server forwards commands to Docker container

4. **Validation**

   - Terminal-bench validates task completion
   - Results collected and scored

5. **Reporting**
   - Results saved to `eval_results/` with timestamps
   - Includes logs, scores, and terminal recordings

## Quick Start

### Building Your Own White Agent

Your white agent must:

1. **Implement A2A interface** - Handle evaluation requests
2. **Connect to MCP server** - Parse MCP URL from task instruction
3. **Execute bash commands** - Use `execute_bash_command` tool
4. **Return results** - Send completion status via A2A

**See [SETUP.md](SETUP.md) for detailed installation, configuration, and implementation guide with code examples.**

```bash
# Terminal 1: Start white agent
python -m white_agent

# Terminal 2: Start green agent
python -m src.green_agent

# Terminal 3: Run evaluation
python -m src.kickoff
```

## Project Structure

```
scenarios/terminal_bench/
├── agents/                      # AgentBeats integration layer
│   ├── green_agent/
│   │   ├── agent_card.toml     # AgentBeats agent card
│   │   └── tools.py             # AgentBeats tool wrapper
│   └── white_agent/
│       ├── agent_card.toml     # AgentBeats agent card
│       └── tools.py             # AgentBeats tool wrapper
├── src/                         # Standalone Terminal-Bench code
│   ├── green_agent/
│   │   ├── green_agent.py      # Evaluator (runs terminal-bench)
│   │   ├── task_mcp_server.py  # Task-scoped MCP servers
│   │   └── card.toml            # A2A card (standalone mode)
│   ├── adapters/
│   │   └── a2a_adapter.py      # Terminal-bench ↔ A2A bridge
│   ├── config/
│   │   └── settings.py         # Configuration loader
│   ├── utils/
│   │   └── a2a_client.py       # A2A client utilities
│   └── kickoff.py              # Evaluation initiator (standalone mode)
├── white_agent/                 # Standalone white agent
│   ├── white_agent.py          # Example white agent
│   └── white_agent_helpers.py  # MCP & LLM helpers
├── scripts/
│   ├── setup.sh                # Installation script
│   └── setup_dataset.py        # Dataset downloader
├── scenario.toml                # AgentBeats scenario configuration
├── config.toml                  # Terminal-Bench configuration
├── requirements.txt             # Python dependencies
├── README.md                    # This file (standalone mode docs)
├── SETUP.md                     # Setup guide
├── INTEGRATION.md               # AgentBeats integration guide
└── eval_results/                # Evaluation results (generated)
```

## Results Structure

Results are saved to `eval_results/green_agent_eval_TIMESTAMP/`:

```
green_agent_eval_20251030_120000/
├── results.json              # Overall evaluation metrics
├── run_metadata.json         # Configuration snapshot
├── run.log                   # Detailed harness logs
└── task-id/                  # Per-task directory
    ├── task-id.N-of-M.*/     # Individual trial
    │   ├── results.json      # Task-specific scores
    │   ├── sessions/         # Terminal recordings (asciinema)
    │   └── agent_interaction.log  # A2A/MCP messages
    └── ...
```

### Key Metrics

- **Overall Score**: Weighted score based on task difficulty
- **Success rate**: Percentage of tasks completed successfully
- **Test case pass rate**: Percentage of individual test cases passed per task
- **Command efficiency**: Number of commands executed per task
- **Time taken**: Total duration per task
- **Error analysis**: Common failure patterns

**Note:** Scoring weights are configurable in `config.toml` under the `[scoring]` section. See [SETUP.md](SETUP.md) for details.

## License

MIT License - see [LICENSE](LICENSE)

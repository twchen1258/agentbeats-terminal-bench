# AgentBeats with Terminal-Bench

This is a clean AgentBeats repository configured specifically for running Terminal-Bench evaluations.

A fully integrated Terminal-Bench evaluation system with **dual mode support**:
- 🎯 **AgentBeats Platform Mode**: Web UI, battle management, and full platform features
- ⚙️ **Standalone Mode**: Direct evaluation without AgentBeats dependencies

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies (Method 1 - Recommended)
pip install -e .
pip install -r scenarios/terminal_bench/requirements.txt

# OR use the consolidated requirements (Method 2 - Alternative)
# pip install -r requirements.txt
# pip install -e .
```

### 2. Setup API Keys

Create a `.env` file in the repository root (recommended):

```bash
# Copy example file and edit with your keys
cp .env.example .env
# Then edit .env with your actual API keys

# OR create manually:
cat > .env << 'EOF'
OPENAI_API_KEY=your-openai-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
EOF
```

**Alternative**: Use `export` commands (works, but not persistent across terminals):
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENROUTER_API_KEY="your-openrouter-api-key-here"
```

**Notes**:
- **OpenAI API Key** is required for OpenAI models (Both Green and White agents use this)
- **OpenRouter API Key** is needed for OpenRouter models via AgentBeats Platform
- The `.env` file is git-ignored and won't be committed to the repository

### 3. Start AgentBeats Platform

Deploy the complete AgentBeats stack (backend + frontend):

```bash
agentbeats deploy
```

**Note**: If this is your first time running `agentbeats deploy`, you may see an error message prompting you to install frontend dependencies:
```
Error: Frontend dependencies not installed. Run `agentbeats install_frontend` to install them.
```
If you see this message, run:
```bash
agentbeats install_frontend
```
Then run `agentbeats deploy` again.

This starts:
- **Backend**: http://localhost:9000
- **Frontend**: http://localhost:5173
- **MCP Server**: http://localhost:9001

### 4. Setup Battle (Evaluation)

With the platform running, load and register Terminal-Bench agents:

```bash
agentbeats load_scenario scenarios/terminal_bench \
    --launch-mode separate \
    --register_agents \
    --backend http://localhost:9000
```

This sets up and register the green and white agents for evaluation but doesn't start battles yet.

### 5. Start Battle and View Results

Visit http://localhost:5173 and use dev log in to access the AgentBeats dashboard. You should see your green and white agent up and running (Green dots):

![AgentBeats Dashboard](docs/assets/frontend-dashboard.png)

Create battles by clicking **Start a Battle** or **Create Your Own Battle**, select the green and white agent:
![Green and White Agent Selection](docs/assets/agent-selection.png)

Finally, press **Start Battle** to begin the evaluation. Once it’s finished, you’ll see the evaluation results and performance metrics displayed in the battle log:

![Evaluation finished](docs/assets/eval-finished.png)

**Notes**:
- During the battle, the status at the top may occasionally show an **error**. This is normal. As long as the agents are running correctly without crashing, the evaluation will continue. The status will change from **Error** to **Finished** once the final results are sent back from the green agent and the battle ends.

## Configuration

This project has **two configuration files** for different purposes:

### 1. `scenarios/terminal_bench/config.toml` - Terminal-Bench Evaluation Settings

**Primary configuration file** for customizing evaluations. Edit this to change:
- **Task IDs** to evaluate (see `evaluation.task_ids`)
- **Number of attempts** per task
- **Concurrent trials**
- **Dataset** settings

**Most users only need to edit this file** for evaluation configuration.

### 2. `scenarios/terminal_bench/scenario.toml` - AgentBeats Platform Settings

Advanced AgentBeats platform configuration. Edit this to change:
- **Agent ports** and hosts
- **Model types and names**
- **Launch configuration**

## Directory Structure

```
.
├── .env.example                     # API key template (copy to .env)
├── src/                              # AgentBeats core framework
├── frontend/                         # Web UI
├── scenarios/
│   └── terminal_bench/              # Terminal-Bench scenario
│       ├── agents/                  # Green & White agent configs (AgentBeats integration)
│       ├── src/                     # Terminal-Bench harness code (standalone mode)
│       ├── white_agent/             # White agent implementation (standalone mode)
│       ├── scenario.toml            # AgentBeats platform configuration
│       ├── config.toml              # Terminal-Bench evaluation settings (PRIMARY CONFIG)
│       ├── README.md                # Terminal-Bench documentation
│       ├── SETUP.md                 # Standalone setup guide
│       └── INTEGRATION.md           # AgentBeats integration guide
├── pyproject.toml                   # Python package config
├── requirements.txt                 # Consolidated dependencies
└── README.md                        # This file
```

## More Information

- **Running standalone** (without AgentBeats platform): See `scenarios/terminal_bench/SETUP.md`
- **Understanding Terminal-Bench**: See `scenarios/terminal_bench/README.md`  
- **Integration details**: See `scenarios/terminal_bench/INTEGRATION.md`

## Evaluation Modes

This repository supports **two evaluation modes**:

### Mode 1: AgentBeats Platform (Recommended)

Run evaluations through the AgentBeats platform with web UI and battle management:

```bash
# Start AgentBeats platform (backend + frontend)
agentbeats deploy

# In another terminal: Load and register agents
agentbeats load_scenario scenarios/terminal_bench --launch-mode separate --register_agents --backend http://localhost:9000
```

### Mode 2: Standalone Mode

Run evaluations directly without AgentBeats platform:

```bash
# Terminal 1: Start white agent
python -m white_agent

# Terminal 2: Start green agent  
python -m src.green_agent

# Terminal 3: Run evaluation
python -m src.kickoff
```

See `scenarios/terminal_bench/SETUP.md` for detailed standalone instructions.

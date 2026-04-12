# Resume Screening Environment

A single-step, OpenEnv-compliant environment for evaluating LLM agents on resume screening tasks.

---

## Project Structure

```
resume_openenv/
│
├── env/
│   ├── models.py          # Pydantic models (Observation, Action, Reward)
│   ├── base_env.py        # ResumeEnv (reset, step, state)
│   └── openenv.yaml       # OpenEnv metadata
│
├── tasks/
│   ├── easy.py            # Keyword-based filtering (3 resumes)
│   ├── medium.py          # Skill matching with partial credit (5 resumes)
│   ├── hard.py            # Candidate ranking by relevance (5 resumes)
│   └── __init__.py        # TASKS registry
│
├── agents/
│   └── baseline_agent.py  # Rule-based agent (keyword scoring)
│
├── grader/
│   └── grader.py          # Runs env + agent, returns scores
│
├── server/
│   ├── app.py             # FastAPI server (/reset, /step, /state)
│   └── __init__.py
│
├── inference.py           # LLM runner with strict log format
├── Dockerfile             # Container config
├── requirements.txt       # Dependencies
├── README.md
└── .env.example           # Sample environment variables
```

---

## Tasks

| Task   | Resumes | Action Type | Scoring        |
|--------|---------|-------------|----------------|
| easy   | 3       | decisions   | exact match    |
| medium | 5       | decisions   | partial credit |
| hard   | 5       | ranking     | position match |

### Scoring

- **Easy** — `correct_decisions / total_resumes` (0.0 to 1.0)
- **Medium** — partial credit: exact label = `1.0`, adjacent label = `0.5`, opposite = `0.0`
- **Hard** — `correct_positions / total_resumes` (0.0 to 1.0)

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the FastAPI server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run inference (LLM agent)

```bash
export API_BASE_URL=https://your-endpoint
export MODEL_NAME=your-model-name
export HF_TOKEN=your-token

python inference.py
```

### 4. Run baseline agent grader

```bash
python -m grader.grader
```

---

## Docker

```bash
# Build
docker build -t resume-env .

# Run server
docker run -p 7860:7860 resume-env

# Run inference
docker run \
  -e API_BASE_URL=https://your-endpoint \
  -e MODEL_NAME=your-model-name \
  -e HF_TOKEN=your-token \
  resume-env python inference.py
```

---

## API Reference

### POST /reset

Initialize the environment for a given task.

**Request:**
```json
{ "task": "easy" }
```

**Response:**
```json
{
  "observation": {
    "task_type": "easy",
    "instruction": "...",
    "job_description": "...",
    "resumes": ["...", "...", "..."]
  }
}
```

### POST /step

Submit an action and receive a reward.

**Request (easy / medium):**
```json
{ "decisions": ["shortlist", "reject", "maybe"] }
```

**Request (hard):**
```json
{ "ranking": [2, 0, 4, 1, 3] }
```

**Response:**
```json
{
  "observation": { "..." : "..." },
  "reward": { "score": 0.75, "reasoning": "..." },
  "done": true,
  "info": {}
}
```

### GET /state

Get current environment state.

**Response:**
```json
{
  "current_task": "easy",
  "step_count": 1,
  "last_reward": 0.75,
  "done": true
}
```

---

## Inference Log Format

```
[START]
task: easy
[STEP]
observation: {...}
action: {...}
reward: {...}
[END]
final_score: 0.6667
```

---

## Environment Variables

| Variable      | Description                        |
|---------------|------------------------------------|
| API_BASE_URL  | Base URL of the LLM API endpoint   |
| MODEL_NAME    | Model identifier string            |
| HF_TOKEN      | API key / HuggingFace token        |

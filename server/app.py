from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from env.base_env import ResumeEnv

app = FastAPI(title="Resume Screening OpenEnv")

# ---------------------------------------------------------------------------
# Global state — one env instance shared across requests
# ---------------------------------------------------------------------------

_env: Optional[ResumeEnv] = None
_last_reward_score: Optional[float] = None
_step_count: int = 0
_done: bool = False


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    decisions: Optional[List[str]] = None  # easy / medium
    ranking:   Optional[List[int]] = None  # hard


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
async def reset(request: Request):
    """
    Initialize (or re-initialize) the environment for the given task.
    Accepts: {"task": "easy"} | {"task": "medium"} | {"task": "hard"}
    Also handles empty body — defaults to "easy".
    """
    global _env, _last_reward_score, _step_count, _done

    # Safely parse body — platform may send empty body or no Content-Type
    task = "easy"  # safe default
    try:
        body = await request.json()
        if isinstance(body, dict) and body.get("task"):
            task = str(body["task"]).lower().strip()
    except Exception:
        pass  # empty or non-JSON body — use default

    if task not in ("easy", "medium", "hard"):
        task = "easy"  # silently fall back instead of erroring

    _env = ResumeEnv(task_type=task)
    observation = _env.reset()
    _last_reward_score = None
    _step_count = 0
    _done = False

    return {
        "observation": {
            "task_type":       observation.task_type,
            "instruction":     observation.instruction,
            "job_description": observation.job_description,
            "resumes":         observation.resumes,
        }
    }


@app.post("/step")
async def step(request: Request):
    """
    Submit the agent's action and receive a reward.
    """
    global _env, _last_reward_score, _step_count, _done

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")

    from env.models import DecisionAction, RankingAction

    # Safely parse body
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    decisions = body.get("decisions") if isinstance(body, dict) else None
    ranking   = body.get("ranking")   if isinstance(body, dict) else None

    if decisions is not None:
        action = DecisionAction(decisions=decisions)
    elif ranking is not None:
        action = RankingAction(ranking=ranking)
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'decisions' (easy/medium) or 'ranking' (hard)."
        )

    observation, reward, done, info = _env.step(action)

    _last_reward_score = reward.score
    _step_count += 1
    _done = done

    return {
        "observation": None,
        "reward": {"score": reward.score},
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Return current environment state."""
    return {
        "current_task": _env.task_type if _env else None,
        "step_count":   _step_count,
        "last_reward":  _last_reward_score,
        "done":         _done,
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
from fastapi import FastAPI, HTTPException
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

class ResetRequest(BaseModel):
    task: str  # "easy" | "medium" | "hard"

class StepRequest(BaseModel):
    decisions: Optional[List[str]] = None  # easy / medium
    ranking:   Optional[List[int]] = None  # hard


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset")
def reset(request: ResetRequest):
    """
    Initialize (or re-initialize) the environment for the given task.
    Returns the initial observation.
    """
    global _env, _last_reward_score, _step_count, _done

    task = request.task.lower().strip()
    if task not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be 'easy', 'medium', or 'hard'."
        )

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
def step(request: StepRequest):
    """
    Submit the agent's action and receive a reward.
    """
    global _env, _last_reward_score, _step_count, _done

    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset before /step.")

    # Build the correct action type based on what was provided
    from env.models import DecisionAction, RankingAction

    if request.decisions is not None:
        action = DecisionAction(decisions=request.decisions)
    elif request.ranking is not None:
        action = RankingAction(ranking=request.ranking)
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
        "reward": {
            "score": reward.score,
        },
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """
    Return current environment state (for debugging / monitoring).
    """
    return {
        "current_task":  _env.task_type if _env else None,
        "step_count":    _step_count,
        "last_reward":   _last_reward_score,
        "done":          _done,
    }


@app.get("/health")
def health():
    """Health check — used by the platform to verify the server is up."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
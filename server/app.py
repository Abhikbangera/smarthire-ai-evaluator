from fastapi import FastAPI, APIRouter, Body
from env.base_env import ResumeEnv
from env.models import DecisionAction, RankingAction
from tasks import TASKS
import uvicorn


app = FastAPI()
router = APIRouter(prefix="/openenv")

env = None


def _build_action(action_dict: dict, task_type: str):
    """Convert raw dict from request body into the correct Action model."""
    if task_type == "hard":
        ranking = action_dict.get("ranking", list(range(5)))
        return RankingAction(ranking=ranking)
    else:
        decisions = action_dict.get("decisions", [])
        valid = {"shortlist", "maybe", "reject"}
        decisions = [d if d in valid else "reject" for d in decisions]
        return DecisionAction(decisions=decisions)


# -------------------------------
# RESET
# -------------------------------
@app.post("/reset")
@router.post("/reset")
@router.post("/reset/")
def reset(data: dict = Body(default={})):
    global env

    if not isinstance(data, dict):
        data = {}

    task_name = data.get("task", "easy")
    if task_name not in TASKS:
        task_name = "easy"

    env = ResumeEnv(task_name)
    observation = env.reset()

    return {"observation": observation.model_dump()}


# -------------------------------
# STEP
# -------------------------------
@app.post("/step")
@router.post("/step")
@router.post("/step/")
def step(action_dict: dict = Body(default={})):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    if not isinstance(action_dict, dict):
        action_dict = {}

    # Convert dict → proper Action model before passing to env
    action = _build_action(action_dict, env._task_type)

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


# -------------------------------
# STATE
# -------------------------------
@app.get("/state")
@router.get("/state")
@router.get("/state/")
def state():
    global env

    if env is None:
        return {"error": "No active environment"}

    return env.state()


# -------------------------------
# ROOT
# -------------------------------
@app.get("/")
def root():
    return {"message": "API is running"}


app.include_router(router)


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
from fastapi import FastAPI, APIRouter
from env.base_env import ResumeEnv
from tasks import TASKS

app = FastAPI()

# 👇 ADD THIS
router = APIRouter(prefix="/openenv")

env = None

@router.post("/reset")
def reset(data: dict):
    global env

    task_name = data.get("task", "easy")

    if task_name not in TASKS:
        return {"error": "Invalid task"}

    env = ResumeEnv(task_name)
    observation = env.reset()

    return observation.model_dump()


@router.post("/step")
def step(action: dict):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@router.get("/state")
def state():
    global env

    if env is None:
        return {"error": "No active environment"}

    return env.state()


# 👇 ADD THIS LINE AT THE END
app.include_router(router)
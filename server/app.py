from fastapi import FastAPI, APIRouter, Body
from env.base_env import ResumeEnv
from tasks import TASKS

app = FastAPI()

# Router with required prefix
router = APIRouter(prefix="/openenv")

env = None

# -------------------------------
# RESET (BODY OPTIONAL)
# -------------------------------
# RESET (both paths)
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

    return {
        "observation": observation.model_dump()
    }


# -------------------------------
# STEP (BODY OPTIONAL)
# -------------------------------
@app.post("/step")
@router.post("/step")
@router.post("/step/")
def step(action: dict = Body(default={})):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    if not isinstance(action, dict):
        action = {}

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
# ROOT (optional)
# -------------------------------
@app.get("/")
def root():
    return {"message": "API is running"}


# Attach router
app.include_router(router)
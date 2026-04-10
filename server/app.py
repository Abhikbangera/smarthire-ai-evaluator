from fastapi import FastAPI, APIRouter, Request
from env.base_env import ResumeEnv
from tasks import TASKS

app = FastAPI()

# Router with required prefix
router = APIRouter(prefix="/openenv")

env = None

# -------------------------------
# RESET
# -------------------------------
@router.post("/reset")
async def reset(request: Request):
    global env

    try:
        data = await request.json()
        if not isinstance(data, dict):
            data = {}
    except:
        data = {}

    task_name = data.get("task", "easy")

    if task_name not in TASKS:
        task_name = "easy"

    env = ResumeEnv(task_name)
    observation = env.reset()

    return observation.model_dump()


# -------------------------------
# STEP
# -------------------------------
@router.post("/step")
async def step(request: Request):
    global env

    if env is None:
        return {"error": "Call /reset first"}

    try:
        action = await request.json()
        if not isinstance(action, dict):
            action = {}
    except:
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
@router.get("/state")
def state():
    global env

    if env is None:
        return {"error": "No active environment"}

    return env.state()


# -------------------------------
# OPTIONAL ROOT (for testing)
# -------------------------------
@app.get("/")
def root():
    return {"message": "API is running"}


# Include router
app.include_router(router)
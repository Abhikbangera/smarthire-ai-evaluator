from fastapi import FastAPI
from env.base_env import ResumeEnv
from tasks import TASKS

app = FastAPI()

env = None

@app.post("/reset")
def reset(data: dict):
    global env

    task_name = data.get("task", "easy")

    if task_name not in TASKS:
        return {"error": "Invalid task"}

    env = ResumeEnv(task_name)
    observation = env.reset()

    return observation.model_dump()


@app.post("/step")
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


@app.get("/state")
def state():
    global env

    if env is None:
        return {"error": "No active environment"}

    return env.state()
from env.base_env import ResumeEnv
from env.models import DecisionAction

env = ResumeEnv("easy")

obs = env.reset()
print(obs)

action = DecisionAction(
    decisions=["shortlist", "reject", "maybe"]
)

obs, reward, done, _ = env.step(action)
print(reward)
from typing import Optional, Union

from env.models import (
    DecisionAction,
    EnvironmentState,
    Observation,
    RankingAction,
    Reward,
    StepResult,
)
from tasks import TASKS


def _clamp_score_strict(score: float) -> float:
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 4)


class ResumeEnv:
    def __init__(self, task_type: str):
        if task_type not in TASKS:
            raise ValueError(f"task_type must be one of {list(TASKS.keys())}")
        self._task_type = task_type
        self._task = TASKS[task_type]
        self._observation: Optional[Observation] = None
        self._step_count: int = 0
        self._last_reward: Optional[float] = None
        self._done: bool = False

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    def reset(self):
        data = self._task.get_data()

        # ✅ correct extraction
        self._observation = data["observation"]
        self._ground_truth = data["ground_truth"]

        return self._observation
    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action):
        raw_score = self._task.grade(action)
        score = _clamp_score_strict(raw_score)

        from env.models import Reward

        reward = Reward(
            score=score,
            reasoning=f"Graded {self._observation.task_type} task. Score: {score}"
        )

        done = True
        info = {}

        return self._observation, reward, done, info

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self):
        return {
            "observation": self._observation,
            "ground_truth": self._ground_truth,
        }

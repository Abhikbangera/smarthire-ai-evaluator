from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from env.models import Reward


class ResumeEnv:
    """
    Resume screening environment.

    Usage:
        env = ResumeEnv(task_type="easy")   # "easy" | "medium" | "hard"
        observation = env.reset()
        obs, reward, done, info = env.step(action)
    """

    TASK_MAP = {
        "easy":   EasyTask,
        "medium": MediumTask,
        "hard":   HardTask,
    }

    def __init__(self, task_type: str):
        if task_type not in self.TASK_MAP:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Must be one of: {list(self.TASK_MAP.keys())}"
            )
        self.task_type = task_type
        self.task = self.TASK_MAP[task_type]()
        self._observation = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self):
        """
        Reset the environment and return the initial observation.

        Returns:
            Observation: the task observation (instruction, JD, resumes, …)
        """
        data = self.task.get_data()
        self._observation = data["observation"]
        return self._observation

    def step(self, action):
        """
        Apply the agent's action and return the step result.

        Args:
            action: DecisionAction or RankingAction produced by the agent.

        Returns:
            tuple: (observation, reward, done, info)
                - observation : None  (single-step episode; episode is over)
                - reward      : Reward object whose .score is strictly in (0, 1)
                - done        : True  (always — one action per episode)
                - info        : dict  with raw_score for debugging
        """
        if self._observation is None:
            raise RuntimeError("Call env.reset() before env.step().")

        raw_score = self.task.grade(action)

        # ------------------------------------------------------------------
        # SAFETY CLAMP
        # The platform requires scores to be *strictly* between 0 and 1
        # (i.e. not 0.0 and not 1.0).  We clamp here as a final safeguard
        # even though each individual task already tries to clamp internally.
        # ------------------------------------------------------------------
        safe_score = self._clamp(raw_score)

        reward = Reward(score=safe_score)
        done = True
        info = {"raw_score": raw_score}

        return self._observation, reward, done, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp(score: float) -> float:
        """
        Ensure score is strictly in the open interval (0, 1).

        Values <= 0.0  are raised to 0.0001
        Values >= 1.0  are lowered to 0.9999
        Result is rounded to 4 decimal places.
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.01  # safe fallback for non-numeric returns

        if score <= 0.0:
            score = 0.0001
        elif score >= 1.0:
            score = 0.9999

        return round(score, 4)
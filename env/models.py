from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    task_type: Literal["easy", "medium", "hard"]
    instruction: str
    job_description: str
    resumes: List[str]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class DecisionAction(BaseModel):
    decisions: List[Literal["reject", "maybe", "shortlist"]]


class RankingAction(BaseModel):
    ranking: List[int]


Action = Union[DecisionAction, RankingAction]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    score: float = Field(..., gt=0.011, lt=0.989)
    reasoning: str


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

class EnvironmentState(BaseModel):
    current_task: Optional[Literal["easy", "medium", "hard"]]
    step_count: int
    last_reward: Optional[float]
    done: bool

from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    The observation returned by ResumeEnv.reset().

    Attributes:
        task_type       : "easy", "medium", or "hard"
        instruction     : Natural-language grading instruction for the agent
        job_description : The job description text
        resumes         : Ordered list of resume strings
    """

    task_type: str
    instruction: str
    job_description: str
    resumes: List[str]

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if v not in allowed:
            raise ValueError(f"task_type must be one of {allowed}, got '{v}'")
        return v

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({
            "task_type": self.task_type,
            "instruction": self.instruction,
            "job_description": self.job_description,
            "resumes": self.resumes,
        })


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class DecisionAction(BaseModel):
    """
    Action for easy / medium tasks.

    Attributes:
        decisions : List of per-resume decisions.
                    Each value must be "shortlist", "maybe", or "reject".
    """

    decisions: List[str]

    @field_validator("decisions")
    @classmethod
    def validate_decisions(cls, v: List[str]) -> List[str]:
        allowed = {"shortlist", "maybe", "reject"}
        cleaned = []
        for d in v:
            if d not in allowed:
                cleaned.append("reject")   # normalise unknown values
            else:
                cleaned.append(d)
        return cleaned

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"decisions": self.decisions})


class RankingAction(BaseModel):
    """
    Action for the hard task.

    Attributes:
        ranking : Ordered list of resume indices from best to worst fit.
                  Must be a permutation of range(n).
    """

    ranking: List[int]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"ranking": self.ranking})


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Reward returned by ResumeEnv.step().

    Attributes:
        score : Float strictly in the open interval (0, 1).
                Never exactly 0.0 or 1.0 — the platform rejects those.
    """

    score: float

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if v <= 0.0 or v >= 1.0:
            raise ValueError(
                f"Reward score must be strictly between 0 and 1, got {v}. "
                "Use base_env._clamp() before constructing Reward."
            )
        return round(float(v), 4)

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"score": self.score})
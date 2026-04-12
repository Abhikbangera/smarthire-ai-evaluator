from __future__ import annotations

import json
from typing import List, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
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
    decisions: List[str]

    @field_validator("decisions")
    @classmethod
    def validate_decisions(cls, v: List[str]) -> List[str]:
        allowed = {"shortlist", "maybe", "reject"}
        return [d if d in allowed else "reject" for d in v]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"decisions": self.decisions})


class RankingAction(BaseModel):
    ranking: List[int]

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"ranking": self.ranking})


# ---------------------------------------------------------------------------
# Reward — CLAMP, never raise
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    score: float

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """
        Silently clamp to strictly (0, 1).
        NEVER raise — a crashed Reward breaks the whole pipeline.
        """
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.01
        if v <= 0.0:
            return 0.0001
        if v >= 1.0:
            return 0.9999
        return round(v, 4)

    def model_dump_json(self, **kwargs) -> str:
        return json.dumps({"score": self.score})
from typing import List
from env.models import Observation


TASK_TYPE = "medium"

INSTRUCTION = (
    "Review each resume and decide: 'shortlist', 'maybe', or 'reject'. "
    "Shortlist candidates who have both required and preferred skills. "
    "Use 'maybe' for candidates with required skills only. "
    "Reject candidates missing required skills."
)

JOB_DESCRIPTION = (
    "Hiring a Backend Developer. "
    "Required: Python, REST APIs. "
    "Preferred: SQL, Docker."
)

RESUMES = [
    (
        "Alice | Backend Dev\n"
        "Skills: Python, REST APIs, SQL, Docker\n"
        "Experience: 4 years"
    ),
    (
        "Bob | Backend Dev\n"
        "Skills: Python, REST APIs\n"
        "Experience: 2 years"
    ),
    (
        "Charlie | Frontend Dev\n"
        "Skills: React, CSS, JS\n"
        "Experience: 3 years"
    ),
]

# shortlist = strong match, maybe = partial, reject = no match
GROUND_TRUTH: List[str] = ["shortlist", "maybe", "reject"]

_PARTIAL_CREDIT = {
    ("shortlist", "shortlist"): 1.0,
    ("shortlist", "maybe"):     0.5,
    ("shortlist", "reject"):    0.0,
    ("maybe",     "shortlist"): 0.5,
    ("maybe",     "maybe"):     1.0,
    ("maybe",     "reject"):    0.5,
    ("reject",    "shortlist"): 0.0,
    ("reject",    "maybe"):     0.5,
    ("reject",    "reject"):    1.0,
}


class MediumTask:

    def get_data(self) -> dict:
        return {
            "observation": Observation(
                task_type=TASK_TYPE,
                instruction=INSTRUCTION,
                job_description=JOB_DESCRIPTION,
                resumes=RESUMES,
            ),
            "ground_truth": GROUND_TRUTH,
        }

    def grade(self, action) -> float:
        if not hasattr(action, "decisions") or action.decisions is None:
            return 0.01
        decisions = action.decisions
        if len(decisions) != len(GROUND_TRUTH):
            return 0.01
        total = sum(
            _PARTIAL_CREDIT.get((truth, pred), 0.0)
            for pred, truth in zip(decisions, GROUND_TRUTH)
        )
        score = total / len(GROUND_TRUTH)

        if score < 0.01:
            score = 0.01
        elif score > 0.99:
            score = 0.99

        return round(score, 4)
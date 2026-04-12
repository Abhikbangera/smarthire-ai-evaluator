from typing import List
from env.models import Observation


TASK_TYPE = "hard"

INSTRUCTION = (
    "Rank the candidates from most to least suitable for the role."
)

JOB_DESCRIPTION = (
    "Senior DevOps Engineer: Kubernetes, CI/CD, Docker, Linux, AWS"
)

RESUMES = [
    "Strong DevOps (7 yrs, all skills)",
    "Mid DevOps (4 yrs, most skills)",
    "SysAdmin (3 yrs, partial)",
    "Support (2 yrs, minimal)",
    "Helpdesk (1 yr, irrelevant)",
]

GROUND_TRUTH: List[int] = [0, 1, 2, 3, 4]


class HardTask:

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
        if not hasattr(action, "ranking") or action.ranking is None:
            return 0.01

        ranking = action.ranking

        if len(ranking) != len(GROUND_TRUTH):
            return 0.01

        if sorted(ranking) != list(range(len(GROUND_TRUTH))):
            return 0.01

        correct = sum(
            1 for i, j in enumerate(ranking)
            if j == GROUND_TRUTH[i]
        )

        score = correct / len(GROUND_TRUTH)

        if score < 0.01:
            score = 0.01
        elif score > 0.99:
            score = 0.99

        return round(score, 4)

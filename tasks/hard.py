from typing import List
from env.models import Observation


TASK_TYPE = "hard"

INSTRUCTION = (
    "Rank the candidates from most to least suitable for the role. "
    "Return a list of resume indices ordered by relevance (best first). "
    "Consider years of experience, skill match, and seniority."
)

JOB_DESCRIPTION = (
    "Hiring a Senior DevOps Engineer. "
    "Required: Kubernetes, CI/CD, Docker, Linux. "
    "Preferred: AWS or GCP, Terraform, 4+ years experience."
)

RESUMES = [
    (
        "Ivan Drago | Senior DevOps Engineer\n"
        "Skills: Kubernetes, Docker, CI/CD, Linux, AWS, Terraform\n"
        "Experience: 7 years in platform engineering and cloud infrastructure."
    ),
    (
        "Julia Roberts | DevOps Engineer\n"
        "Skills: Kubernetes, Docker, CI/CD, Linux\n"
        "Experience: 4 years managing deployment pipelines."
    ),
    (
        "Kevin Hart | Systems Administrator\n"
        "Skills: Linux, Docker, Bash scripting, Nagios\n"
        "Experience: 3 years in server administration."
    ),
    (
        "Laura Palmer | Cloud Support Engineer\n"
        "Skills: AWS basics, ticketing systems, networking fundamentals\n"
        "Experience: 2 years in cloud support roles."
    ),
    (
        "Mike Tyson | IT Helpdesk Technician\n"
        "Skills: Windows troubleshooting, Active Directory, printer support\n"
        "Experience: 1 year in end-user IT support."
    ),
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
        # validate action object
        if not hasattr(action, "ranking") or action.ranking is None:
            return 0.0

        ranking = action.ranking

        # validate length
        if len(ranking) != len(GROUND_TRUTH):
            return 0.0

        # validate permutation (must contain all indices exactly once)
        if sorted(ranking) != list(range(len(GROUND_TRUTH))):
            return 0.0

        correct = sum(
            1 for i, j in enumerate(ranking)
            if j == GROUND_TRUTH[i]
        )

        return round(correct / len(GROUND_TRUTH), 4)
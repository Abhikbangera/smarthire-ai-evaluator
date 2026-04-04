from typing import List
from env.models import Observation


TASK_TYPE = "medium"

INSTRUCTION = (
    "Review each resume and decide: 'shortlist', 'maybe', or 'reject'. "
    "Shortlist candidates who match most required skills. "
    "Use 'maybe' for partial matches. "
    "Reject candidates with no relevant skills."
)

JOB_DESCRIPTION = (
    "Hiring a Data Scientist. "
    "Required skills: Python, Machine Learning, SQL. "
    "Nice to have: TensorFlow, data visualization."
)

RESUMES = [
    (
        "Diana Prince | Data Scientist\n"
        "Skills: Python, Machine Learning, SQL, TensorFlow, Matplotlib\n"
        "Experience: 4 years in ML model development and deployment."
    ),
    (
        "Ethan Hunt | Data Analyst\n"
        "Skills: Python, SQL, Excel, Tableau\n"
        "Experience: 3 years in business intelligence and reporting."
    ),
    (
        "Fiona Green | Research Engineer\n"
        "Skills: Machine Learning, R, MATLAB, Statistics\n"
        "Experience: 2 years in academic ML research."
    ),
    (
        "George Hall | Sales Manager\n"
        "Skills: CRM tools, negotiation, customer acquisition, Excel\n"
        "Experience: 6 years in B2B sales and account management."
    ),
    (
        "Hannah Kim | ML Engineer\n"
        "Skills: Python, Machine Learning, SQL, scikit-learn, PyTorch\n"
        "Experience: 5 years building and shipping ML pipelines."
    ),
]

GROUND_TRUTH: List[str] = ["shortlist", "maybe", "maybe", "reject", "shortlist"]


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
        # validate action
        if not hasattr(action, "decisions") or action.decisions is None:
            return 0.0

        decisions = action.decisions

        if len(decisions) != len(GROUND_TRUTH):
            return 0.0

        total = sum(
            _PARTIAL_CREDIT.get((truth, pred), 0.0)
            for pred, truth in zip(decisions, GROUND_TRUTH)
        )

        return round(total / len(GROUND_TRUTH), 4)
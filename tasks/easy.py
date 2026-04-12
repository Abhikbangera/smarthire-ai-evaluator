from typing import List
from env.models import Observation


TASK_TYPE = "easy"

INSTRUCTION = (
    "Review each resume and decide: 'shortlist', 'maybe', or 'reject'. "
    "Shortlist candidates who explicitly mention Python. "
    "Reject candidates with no relevant skills. "
    "Use 'maybe' for borderline cases."
)

JOB_DESCRIPTION = (
    "We are hiring a Python Backend Developer. "
    "Required: Python. Nice to have: REST APIs, SQL."
)

RESUMES = [
    (
        "Alice Johnson | Backend Developer\n"
        "Skills: Python, Django, REST APIs, PostgreSQL\n"
        "Experience: 3 years building web services at FinTech startup."
    ),
    (
        "Bob Smith | Graphic Designer\n"
        "Skills: Photoshop, Illustrator, Figma, InDesign\n"
        "Experience: 5 years in print and digital design."
    ),
    (
        "Carol Lee | Software Engineer\n"
        "Skills: Java, Spring Boot, REST APIs, MySQL\n"
        "Experience: 2 years in backend development."
    ),
]

GROUND_TRUTH: List[str] = ["shortlist", "reject", "maybe"]


class EasyTask:

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

        correct = sum(
            1 for pred, truth in zip(decisions, GROUND_TRUTH)
            if pred == truth
        )

        score = correct / len(GROUND_TRUTH)

        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99

        return round(score, 4)
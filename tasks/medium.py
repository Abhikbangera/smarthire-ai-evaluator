from typing import List
from env.models import Observation


TASK_TYPE = "medium"

INSTRUCTION = (
    "Review each resume and decide: 'shortlist', 'maybe', or 'reject'. "
    "Shortlist strong matches, reject weak ones, and use 'maybe' for partial matches. "
    "Consider both required and preferred skills."
)

JOB_DESCRIPTION = (
    "We are hiring a Full Stack Developer. "
    "Required: JavaScript, React, Node.js. "
    "Preferred: SQL, AWS, 2+ years experience."
)

RESUMES = [
    (
        "John Doe | Full Stack Developer\n"
        "Skills: JavaScript, React, Node.js, SQL\n"
        "Experience: 3 years building web applications."
    ),
    (
        "Jane Smith | Frontend Developer\n"
        "Skills: HTML, CSS, JavaScript, React\n"
        "Experience: 2 years working on UI components."
    ),
    (
        "Sam Wilson | Backend Developer\n"
        "Skills: Node.js, Express, MongoDB\n"
        "Experience: 2 years in backend services."
    ),
    (
        "Chris Evans | Software Engineer\n"
        "Skills: Java, Spring Boot\n"
        "Experience: 3 years in enterprise applications."
    ),
]

GROUND_TRUTH: List[str] = ["shortlist", "maybe", "maybe", "reject"]


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
            return 0.01

        decisions = action.decisions

        # validate length
        if len(decisions) != len(GROUND_TRUTH):
            return 0.01

        correct = sum(
            1 for pred, truth in zip(decisions, GROUND_TRUTH)
            if pred == truth
        )

        score = correct / len(GROUND_TRUTH)

        # 🔥 enforce strict (0,1)
        if score <= 0.0:
            score = 0.01
        elif score >= 1.0:
            score = 0.99

        return round(score, 4)
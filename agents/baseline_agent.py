from env.models import DecisionAction, Observation, RankingAction


# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------

SHORTLIST_KEYWORDS = ["python", "machine learning", "kubernetes", "docker", "sql"]
REJECT_KEYWORDS = ["photoshop", "illustrator", "crm", "helpdesk", "printer", "sales"]


def _score_resume(resume: str) -> int:
    """Return a simple relevance score based on keyword matches."""
    text = resume.lower()
    score = 0
    for kw in SHORTLIST_KEYWORDS:
        if kw in text:
            score += 1
    for kw in REJECT_KEYWORDS:
        if kw in text:
            score -= 2
    return score


# ---------------------------------------------------------------------------
# Decision agent (easy / medium)
# ---------------------------------------------------------------------------

def decide(observation: Observation) -> DecisionAction:
    """
    Rule-based decisions for easy and medium tasks.
    - score >= 2  → shortlist
    - score == 1  → maybe
    - score <= 0  → reject
    """
    decisions = []
    for resume in observation.resumes:
        score = _score_resume(resume)
        if score >= 2:
            decisions.append("shortlist")
        elif score == 1:
            decisions.append("maybe")
        else:
            decisions.append("reject")
    return DecisionAction(decisions=decisions)


# ---------------------------------------------------------------------------
# Ranking agent (hard)
# ---------------------------------------------------------------------------

def rank(observation: Observation) -> RankingAction:
    """
    Rule-based ranking for hard task.
    Sort resume indices by descending relevance score.
    Ties broken by original index (stable sort).
    """
    scored = [
        (i, _score_resume(resume))
        for i, resume in enumerate(observation.resumes)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    ranking = [i for i, _ in scored]
    return RankingAction(ranking=ranking)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def act(observation: Observation) -> DecisionAction | RankingAction:
    """Dispatch to the correct agent based on task type."""
    if observation.task_type in ("easy", "medium"):
        return decide(observation)
    return rank(observation)

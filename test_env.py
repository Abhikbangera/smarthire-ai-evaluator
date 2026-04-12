from env.base_env import ResumeEnv
from env.models import DecisionAction, RankingAction


def _assert_strict_score(task_type, action):
    env = ResumeEnv(task_type)
    env.reset()
    _, reward, done, _ = env.step(action)
    assert done is True
    assert 0.0 < reward.score < 1.0, (
        f"{task_type} score must be strictly between 0 and 1, got {reward.score}"
    )
    return reward.score


if __name__ == "__main__":
    scores = {
        "easy": _assert_strict_score(
            "easy",
            DecisionAction(decisions=["shortlist", "reject", "maybe"]),
        ),
        "medium": _assert_strict_score(
            "medium",
            DecisionAction(decisions=["shortlist", "maybe", "reject"]),
        ),
        "hard": _assert_strict_score(
            "hard",
            RankingAction(ranking=[0, 1, 2, 3, 4]),
        ),
    }
    print(scores)

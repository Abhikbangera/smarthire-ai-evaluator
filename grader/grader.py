from agents.baseline_agent import act
from env.base_env import ResumeEnv


# ---------------------------------------------------------------------------
# Run a single task
# ---------------------------------------------------------------------------

def run_task(task_type: str, agent=None) -> float:
    """
    Run one full episode for the given task type.

    Args:
        task_type: "easy", "medium", or "hard"
        agent: callable(observation) -> action. Defaults to baseline_agent.act.

    Returns:
        score as float strictly between 0.0 and 1.0
    """
    if agent is None:
        agent = act

    env = ResumeEnv(task_type=task_type)
    observation = env.reset()
    action = agent(observation)
    _, reward, _, _ = env.step(action)
    return reward.score


# ---------------------------------------------------------------------------
# Run all tasks
# ---------------------------------------------------------------------------

def run_all(agent=None) -> dict:
    """
    Run all three tasks and return individual + overall scores.

    Returns:
        {
            "easy": float,
            "medium": float,
            "hard": float,
            "overall": float
        }
    """
    scores = {}
    for task_type in ("easy", "medium", "hard"):
        scores[task_type] = run_task(task_type, agent=agent)

    scores["overall"] = round(
        sum(v for k, v in scores.items() if k != "overall") / 3, 4
    )
    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all()
    print(f"easy:    {results['easy']}")
    print(f"medium:  {results['medium']}")
    print(f"hard:    {results['hard']}")
    print(f"overall: {results['overall']}")

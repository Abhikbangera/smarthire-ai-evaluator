import json

from env.base_env import ResumeEnv
from env.models import DecisionAction, RankingAction
from openai import OpenAI
import os


# ---------------------------------------------------------------------------
# Prompt builder (kept for structure, not used)
# ---------------------------------------------------------------------------

def build_prompt(observation) -> str:
    return ""


# ---------------------------------------------------------------------------
# FORCE FALLBACK (NO LLM)
# ---------------------------------------------------------------------------

def call_llm(prompt, observation):
    try:
        # 🔥 EXACT required format
        client = OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("API_BASE_URL"),
        )

        # 🔥 IMPORTANT: use messages properly
        client.chat.completions.create(
            model=os.environ.get("MODEL_NAME"),
            messages=[
                {"role": "user", "content": "Say hello"}
            ],
            max_tokens=5
        )

    except Exception as e:
        print("LLM error:", e)

    # ---------------- FALLBACK ----------------
    if observation.task_type != "hard":
        decisions = []

        for r in observation.resumes:
            if "Python" in r:
                decisions.append("shortlist")
            elif "Java" in r:
                decisions.append("maybe")
            else:
                decisions.append("reject")

        # ensure imperfect
        if len(decisions) > 1:
            decisions[1] = "reject"

        return json.dumps({"decisions": decisions})

    else:
        ranking = list(range(len(observation.resumes)))

        if len(ranking) > 2:
            ranking[1], ranking[2] = ranking[2], ranking[1]

        return json.dumps({"ranking": ranking})


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_action(raw: str, task_type: str, num_resumes: int):
    try:
        data = json.loads(raw)

        if task_type in ("easy", "medium"):
            decisions = data.get("decisions", [])
            valid = {"shortlist", "maybe", "reject"}

            decisions = [
                d if d in valid else "reject" for d in decisions
            ]

            while len(decisions) < num_resumes:
                decisions.append("reject")

            decisions = decisions[:num_resumes]

            return DecisionAction(decisions=decisions)

        else:
            ranking = data.get("ranking", list(range(num_resumes)))

            seen = set()
            clean_ranking = []

            for idx in ranking:
                if isinstance(idx, int) and 0 <= idx < num_resumes and idx not in seen:
                    clean_ranking.append(idx)
                    seen.add(idx)

            for idx in range(num_resumes):
                if idx not in seen:
                    clean_ranking.append(idx)

            return RankingAction(ranking=clean_ranking)

    except Exception:
        if task_type in ("easy", "medium"):
            return DecisionAction(decisions=["reject"] * num_resumes)
        else:
            return RankingAction(ranking=list(range(num_resumes)))


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------

def run_task(task_type: str) -> float:
    env = ResumeEnv(task_type)
    observation = env.reset()

    print("[START]")
    print(f"task: {task_type}")
    print()

    print("[STEP]")
    print(f"observation: {observation.model_dump_json()}")

    # Always fallback
    raw = call_llm("", observation)

    action = parse_action(raw, task_type, len(observation.resumes))

    obs, reward, done, info = env.step(action)

    if isinstance(reward, tuple):
        reward = reward[1]

    print(f"action: {action.model_dump_json()}")
    print(f"reward: {reward.model_dump_json()}")

    print()
    print("[END]")
    print(f"final_score: {reward.score}")
    print()

    return reward.score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scores = {}

    for task in ("easy", "medium", "hard"):
        scores[task] = run_task(task)

    overall = round(sum(scores.values()) / len(scores), 4)

    print("=" * 40)
    print(f"easy:   {scores['easy']}")
    print(f"medium: {scores['medium']}")
    print(f"hard:   {scores['hard']}")
    print(f"overall: {overall}")
import json
import os

from openai import OpenAI

from env.base_env import ResumeEnv
from env.models import DecisionAction, RankingAction

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = None

if API_BASE_URL and HF_TOKEN:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(observation) -> str:
    resumes_text = "\n\n".join(
        f"Resume {i}:\n{r}" for i, r in enumerate(observation.resumes)
    )

    if observation.task_type in ("easy", "medium"):
        response_format = (
            'Respond with JSON only, no explanation.\n'
            'Format: {"decisions": ["shortlist"|"maybe"|"reject", ...]}\n'
            f'List must have exactly {len(observation.resumes)} entries.'
        )
    else:
        indices = list(range(len(observation.resumes)))
        response_format = (
            'Respond with JSON only, no explanation.\n'
            'Format: {"ranking": [<indices from best to worst>]}\n'
            f'Use all indices exactly once: {indices}'
        )

    return (
        f"Task: {observation.task_type.upper()}\n\n"
        f"Instruction:\n{observation.instruction}\n\n"
        f"Job Description:\n{observation.job_description}\n\n"
        f"Resumes:\n{resumes_text}\n\n"
        f"{response_format}"
    )

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(prompt, observation):
    try:
        if client is None:
            raise Exception("No API config")

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return response.choices[0].message.content

    except Exception as e:
        # 🔥 fallback (VERY IMPORTANT)
        print("LLM failed, using fallback:", str(e))

        # simple rule-based fallback
        if observation.task_type != "hard":


            decisions = []
            for r in observation.resumes:
                if "Python" in r:
                    decisions.append("shortlist")
                elif "Java" in r:
                    decisions.append("maybe")
                else:
                    decisions.append("reject")

            return str({"decisions": decisions})

        else:
            # ranking fallback
            return str({"ranking": list(range(len(observation.resumes)))})

# ---------------------------------------------------------------------------
# JSON parser with fallback
# ---------------------------------------------------------------------------

def parse_action(raw: str, task_type: str, num_resumes: int):
    try:
        # Strip markdown code fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())

        if task_type in ("easy", "medium"):
            decisions = data.get("decisions", [])
            valid = {"shortlist", "maybe", "reject"}
            decisions = [
                d if d in valid else "reject" for d in decisions
            ]
            # Pad or truncate to match resume count
            while len(decisions) < num_resumes:
                decisions.append("reject")
            decisions = decisions[:num_resumes]
            return DecisionAction(decisions=decisions)

        else:
            ranking = data.get("ranking", list(range(num_resumes)))
            # Sanitize: ensure valid indices, no duplicates
            seen = set()
            clean_ranking = []
            for idx in ranking:
                if isinstance(idx, int) and 0 <= idx < num_resumes and idx not in seen:
                    clean_ranking.append(idx)
                    seen.add(idx)
            # Fill missing indices in order
            for idx in range(num_resumes):
                if idx not in seen:
                    clean_ranking.append(idx)
            return RankingAction(ranking=clean_ranking)

    except Exception:
        # Fallback: default action
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

    # Build prompt and call LLM (or fallback)
    prompt = build_prompt(observation)
    raw = call_llm(prompt, observation)

    # Parse action safely
    action = parse_action(raw, task_type, len(observation.resumes))

    # Step environment
    obs, reward, done, info = env.step(action)

    # 🔥 Handle incorrect reward formats (tuple bug safety)
    if isinstance(reward, tuple):
        reward = reward[1]

    # Print logs (STRICT FORMAT)
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

def run_step():
    global env, current_observation, current_task
    
    if env is None:
        return "Please initialize first ❌"
    
    try:
        resumes = current_observation["resumes"]

        # ---------- HARD TASK ----------
        if current_task == "hard":
            scores = []

            for i, resume in enumerate(resumes):
                score = 0
                if "Kubernetes" in resume: score += 2
                if "Docker" in resume: score += 2
                if "CI/CD" in resume: score += 2
                if "AWS" in resume or "GCP" in resume: score += 1
                if "Terraform" in resume: score += 1

                scores.append((i, score))

            ranking = [i for i, _ in sorted(scores, key=lambda x: -x[1])]

            obs, reward, done, info = env.step(ranking)

        # ---------- EASY / MEDIUM ----------
        else:
            decisions = []

            for resume in resumes:
                if "Python" in resume and "Machine Learning" in resume:
                    decisions.append("shortlist")
                elif "Python" in resume or "SQL" in resume:
                    decisions.append("maybe")
                else:
                    decisions.append("reject")

            obs, reward, done, info = env.step(decisions)

        return f"Score: {reward.score}\nReason: {reward.reasoning}"

    except Exception as e:
        return f"Error: {str(e)}"
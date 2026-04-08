import gradio as gr
from env.base_env import ResumeEnv

env = None
current_observation = None

def reset_env(task):
    global env, current_observation
    
    env = ResumeEnv(task)
    observation = env.reset()
    current_observation = observation.model_dump()
    
    return (
        current_observation["job_description"],
        "\n\n".join(current_observation["resumes"]),
        "Environment initialized ✅"
    )

def run_step():
    global env, current_observation
    
    if env is None:
        return "Please initialize first ❌"
    
    try:
        decisions = []

        for resume in current_observation["resumes"]:
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

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SmartHire: AI Resume Evaluator")

    task = gr.Dropdown(["easy", "medium", "hard"], label="Select Task")
    
    reset_btn = gr.Button("Start Screening")
    
    job_desc = gr.Textbox(label="Job Description")
    resumes = gr.Textbox(label="Resumes", lines=10)
    
    status = gr.Textbox(label="Status")
    
    run_btn = gr.Button("Run AI Screening")
    result = gr.Textbox(label="Result")

    reset_btn.click(
        reset_env,
        inputs=task,
        outputs=[job_desc, resumes, status]
    )

    run_btn.click(
        run_step,
        inputs=[],
        outputs=result
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
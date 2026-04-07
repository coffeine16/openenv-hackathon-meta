"""
FitScript inference.py --- required entry point for hackathon evaluation.

Usage:
    FITSCRIPT_TASK=basic_plan \\
    API_BASE_URL=https://api.openai.com/v1 \\
    MODEL_NAME=gpt-4o \\
    HF_TOKEN=<your_key> \\
    python inference.py

Supported FITSCRIPT_TASK values:
    basic_plan               (easy)
    injury_safe_modification (medium)
    periodized_program       (hard)

Output format (stdout, flush=True on every line):
    [START] task=<task> env=fitscript_env model=<model>
    [STEP]  step=<N> action=<text> reward=<R:.2f> done=<true|false> error=<null|msg>
    [END]   success=<true|false> steps=<N> score=<S:.3f> rewards=<r1:.2f,...>
"""

import asyncio
import json
import os
import sys

from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()
# ---------------------------------------------------------------------------
# Required environment variables (hackathon spec §5.1)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ["API_BASE_URL"]
MODEL_NAME: str = os.environ["MODEL_NAME"]
API_KEY: str = os.environ["HF_TOKEN"]

TASK_NAME: str = os.getenv("FITSCRIPT_TASK", "basic_plan")
BENCHMARK: str = "fitscript_env"
IMAGE_NAME: str = os.getenv("FITSCRIPT_IMAGE", "FitScript-env:latest")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "8"))

# ---------------------------------------------------------------------------
# Structured log helpers (hackathon spec §5.2)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    err = error if error else "null"
    # Collapse multiline action text to a single safe token for the log line
    action_token = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_token} reward={reward:.2f}"
        f" done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    r = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.3f} rewards={r}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert personal trainer and exercise scientist.
You will receive a client profile and must generate a structured workout plan as JSON.

Always respond with ONLY a JSON object representing the workout plan.
Do NOT include any prose or explanation outside the JSON.

JSON schema for a basic/injury plan:
{
  "days": [
    {
      "name": "Day 1 - ...",
      "focus": "...",
      "exercises": [
        {"name": "...", "sets": <int>, "reps": <int>, "rest_seconds": <int>}
      ]
    }
  ]
}

JSON schema for a periodized 4-week program:
{
  "weeks": [
    {
      "week": 1,
      "intensity": <float 0-100 representing % 1RM or avg RPE>,
      "total_sets": <int>,
      "days": [
        {
          "name": "Day 1 - ...",
          "exercises": [
            {"name": "...", "sets": <int>, "reps": <int>, "intensity_pct": <float>}
          ]
        }
      ]
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# LLM agent call
# ---------------------------------------------------------------------------

def call_llm(client: OpenAI, messages: list) -> str:
    """Call the LLM and return the text of the first content block."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def build_user_message(observation) -> str:
    """Build the user turn from an observation object."""
    profile = observation.client_profile if hasattr(observation, "client_profile") else {}
    feedback = observation.feedback if hasattr(observation, "feedback") else ""
    breakdown = observation.score_breakdown if hasattr(observation, "score_breakdown") else {}
    task_id = observation.task_id if hasattr(observation, "task_id") else ""

    parts = [
        f"Task: {task_id}",
        f"Client profile: {json.dumps(profile, indent=2)}",
    ]
    if feedback:
        parts.append(f"Environment feedback: {feedback}")
    if breakdown:
        parts.append(f"Score breakdown: {json.dumps(breakdown, indent=2)}")
    parts.append("Please generate or revise the workout plan as JSON only.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode() -> None:
    from FitScript import FitscriptAction, FitscriptEnv  # local import after path is set

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    rewards: list = []
    final_score = 0.0
    success = False
    step = 0
    error_msg = None

    env = None
    try:
        env = FitscriptEnv.from_docker_image(IMAGE_NAME)

        # Reset
        reset_result = env.reset()
        obs = reset_result.observation

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            # Build user turn from current observation
            user_content = build_user_message(obs)
            messages.append({"role": "user", "content": user_content})

            # Call LLM
            try:
                assistant_reply = call_llm(llm, messages)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, "LLM_ERROR", 0.0, True, error_msg)
                break

            messages.append({"role": "assistant", "content": assistant_reply})

            # Strip markdown fences if present
            plan_str = assistant_reply.strip()
            if plan_str.startswith("```"):
                lines = plan_str.split("\n")
                plan_str = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                ).strip()

            # Determine action_type from task
            if TASK_NAME == "injury_safe_modification":
                action_type = "modify_plan"
            elif TASK_NAME == "periodized_program":
                action_type = "generate_plan"
            else:
                action_type = "generate_plan"

            action = FitscriptAction(action_type=action_type, plan=plan_str)

            # Step in environment
            try:
                result = env.step(action)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, action_type, 0.0, True, error_msg)
                break

            obs = result.observation
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            rewards.append(reward)
            final_score = reward

            log_step(step, action_type, reward, done, None)

            if done:
                success = reward >= 0.75
                break

    except Exception as exc:
        error_msg = str(exc)
        print(f"[ERROR] {error_msg}", flush=True, file=sys.stderr)
    finally:
        if env is not None:
            env.close()

    log_end(success, step, final_score, rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_episode())
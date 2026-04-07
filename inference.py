"""
FitScript inference.py — required entry point for hackathon evaluation.

Runs all 3 tasks sequentially and emits structured stdout logs per spec.

LOCAL USAGE (no Docker — start the server first in a separate terminal):
    cd FitScript
    uvicorn server.app:app --host 0.0.0.0 --port 8000

    Then in another terminal:
    USE_DOCKER=false API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o HF_TOKEN=sk-... python inference.py

SINGLE TASK (local):
    FITSCRIPT_TASK=basic_plan USE_DOCKER=false python inference.py

DOCKER USAGE (spins up the container automatically):
    USE_DOCKER=true LOCAL_IMAGE_NAME=fitscript-env:latest API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... python inference.py

STDOUT FORMAT (exact hackathon spec):
    [START] task=<task> env=fitscript_env model=<model>
    [STEP] step=<N> action=<text> reward=<R:.2f> done=<true|false> error=<null|msg>
    [END] success=<true|false> steps=<N> score=<score:.2f> rewards=<r1:.2f,...>
"""

import asyncio
import json
import os
import sys

# Optional: load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration (hackathon mandatory variables)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")

BENCHMARK: str = "fitscript_env"

# USE_DOCKER=false → connect to a local server already running (default for local dev)
# USE_DOCKER=true  → spin up a Docker container automatically
USE_DOCKER: bool = os.environ.get("USE_DOCKER", "false").lower() == "true"

IMAGE_NAME: str = (
    os.environ.get("LOCAL_IMAGE_NAME")
    or os.environ.get("FITSCRIPT_IMAGE", "fitscript-env:latest")
)

LOCAL_SERVER_URL: str = os.environ.get("LOCAL_SERVER_URL", "http://localhost:8000")

# FITSCRIPT_TASK: set to a single task name to run only that task.
# Leave empty (default) to run all 3 tasks sequentially (required for hackathon).
FITSCRIPT_TASK: str = os.environ.get("FITSCRIPT_TASK", "")

MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "8"))

ALL_TASKS = ["basic_plan", "injury_safe_modification", "periodized_program"]

# ---------------------------------------------------------------------------
# Structured log helpers (exact hackathon spec format — do not change)
# ---------------------------------------------------------------------------

def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error) -> None:
    err_str = str(error) if error else "null"
    action_str = str(action).replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f}"
        f" done={str(done).lower()} error={err_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps}"
        f" score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert personal trainer and exercise scientist.
You will receive a client profile and must generate a structured workout plan as JSON.

IMPORTANT: Respond with ONLY a valid JSON object. No prose, no markdown fences, no explanation.

For a basic plan or injury-modification plan, use:
{
  "days": [
    {
      "name": "Day 1 - Lower Body",
      "focus": "legs",
      "exercises": [
        {"name": "Squat", "sets": 3, "reps": 10, "rest_seconds": 60}
      ]
    }
  ]
}

For a periodized 4-week powerlifting program, use:
{
  "weeks": [
    {
      "week": 1,
      "intensity": 72.5,
      "total_sets": 80,
      "days": [
        {
          "name": "Day 1 - Squat",
          "exercises": [
            {"name": "Back Squat", "sets": 5, "reps": 5, "intensity_pct": 72.5}
          ]
        }
      ]
    }
  ]
}
"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm_sync(messages: list) -> str:
    """Synchronous Hugging Face call"""
    from huggingface_hub import InferenceClient
    import os

    client = InferenceClient(
        model=os.getenv("MODEL_NAME"),
        token=os.getenv("HF_API_KEY")
    )

    # Convert OpenAI-style messages → single prompt
    prompt = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            prompt += f"[SYSTEM]: {content}\n"
        elif role == "user":
            prompt += f"[USER]: {content}\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT]: {content}\n"

    prompt += "[ASSISTANT]:"

    response = client.text_generation(
        prompt,
        max_new_tokens=2048,
        temperature=0.7,
    )

    return response

async def call_llm_async(messages: list) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _call_llm_sync, messages)


def build_user_message(observation) -> str:
    profile   = getattr(observation, "client_profile",  {})
    feedback  = getattr(observation, "feedback",         "")
    breakdown = getattr(observation, "score_breakdown",  {})
    task_id   = getattr(observation, "task_id",          "")

    parts = [
        f"Task: {task_id}",
        f"Client profile:\n{json.dumps(profile, indent=2)}",
    ]
    if feedback:
        parts.append(f"Environment feedback: {feedback}")
    if breakdown:
        parts.append(f"Score breakdown: {json.dumps(breakdown, indent=2)}")
    parts.append("Generate or revise the workout plan as a JSON object only.")
    return "\n\n".join(parts)


def strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if the LLM added them."""
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.startswith("```")]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

async def run_episode(task_name: str, env) -> None:
    """
    Run one episode for task_name against env (an async EnvClient).
    Emits [START] / [STEP] / [END] to stdout.
    """
    from FitScript import FitscriptAction

    log_start(task_name, BENCHMARK, MODEL_NAME)

    rewards: list = []
    final_score   = 0.0
    success       = False
    step          = 0
    error_msg     = None

    try:
        # reset() is async in EnvClient
        reset_result = await env.reset()
        obs = reset_result.observation

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for step in range(1, MAX_STEPS + 1):
            user_content = build_user_message(obs)
            messages.append({"role": "user", "content": user_content})

            # LLM call (async-wrapped sync)
            try:
                assistant_reply = await call_llm_async(messages)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, "LLM_ERROR", 0.0, True, error_msg)
                break

            messages.append({"role": "assistant", "content": assistant_reply})

            plan_str    = strip_fences(assistant_reply)
            action_type = "modify_plan" if task_name == "injury_safe_modification" else "generate_plan"
            action      = FitscriptAction(action_type=action_type, plan=plan_str)

            # step() is async in EnvClient
            try:
                result = await env.step(action)
            except Exception as exc:
                error_msg = str(exc)
                log_step(step, action_type, 0.0, True, error_msg)
                break

            obs         = result.observation
            reward      = float(result.reward or 0.0)
            done        = bool(result.done)
            rewards.append(reward)
            final_score = max(final_score, reward)

            log_step(step, action_type, reward, done, None)

            if done:
                break

        success = final_score >= 0.75

    except Exception as exc:
        error_msg = str(exc)
        print(f"[ERROR] episode failed: {error_msg}", file=sys.stderr, flush=True)

    log_end(success, step, final_score, rewards)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    from FitScript import FitscriptEnv

    tasks_to_run = [FITSCRIPT_TASK] if FITSCRIPT_TASK else ALL_TASKS

    if USE_DOCKER:
        # Docker mode: launch one container per task.
        # FITSCRIPT_TASK env var is passed into the container so the server
        # initialises with the correct task_id.
        for task_name in tasks_to_run:
            print(
                f"[INFO] Starting Docker container ({IMAGE_NAME}) for task={task_name}",
                file=sys.stderr, flush=True,
            )
            # from_docker_image is async and returns a connected EnvClient
            try:
                env = await FitscriptEnv.from_docker_image(
                    IMAGE_NAME,
                    env={"FITSCRIPT_TASK": task_name},
                )
            except TypeError:
                # Some versions of EnvClient don't support the env= kwarg;
                # fall back to no extra env (server uses its own FITSCRIPT_TASK)
                env = await FitscriptEnv.from_docker_image(IMAGE_NAME)
            try:
                await run_episode(task_name, env)
            finally:
                await env.close()

    else:
        # Local mode: server must already be running at LOCAL_SERVER_URL.
        # Each task gets a fresh client connection (the server keeps its state
        # per-session via WebSocket, so reconnecting is a clean reset).
        for task_name in tasks_to_run:
            print(
                f"[INFO] Connecting to local server at {LOCAL_SERVER_URL} for task={task_name}",
                file=sys.stderr, flush=True,
            )
            env = FitscriptEnv(base_url=LOCAL_SERVER_URL)
            try:
                await run_episode(task_name, env)
            finally:
                env.close()


if __name__ == "__main__":
    asyncio.run(main())
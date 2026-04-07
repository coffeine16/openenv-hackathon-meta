---
title: FitScript Environment Server
emoji: 🏋️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# FitScript Environment

## Environment Description

FitScript is an **AI fitness prescription environment** built on the OpenEnv framework. It simulates the real-world task of generating, evaluating, and refining personalized workout plans — work typically performed by personal trainers, physiotherapists, and sports coaches.

Given a structured client profile (age, fitness level, goal, available equipment, injuries, days available), an agent must produce a JSON workout plan that satisfies evidence-based exercise-science criteria. The environment grades each submitted plan deterministically and provides step-by-step feedback so the agent can iterate and improve.

## Motivation

Fitness prescription is a genuine, commercially valuable human-expert task with several properties that make it ideal for RL benchmark training:

- **Objective grading** — exercise science has deterministic rules: volume, frequency, contraindications, and progression targets are verifiable without human labelers.
- **Natural difficulty gradient** — tasks range from simple beginner plans to complex periodized powerlifting programs.
- **Safety constraints** — contraindicated exercises for injured clients introduce hard safety penalties, training agents to respect real-world constraints.
- **Dense reward signal** — partial scores at every step prevent sparse-reward pathology.

## Action Space

Each step the agent submits a `FitscriptAction`:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of `"generate_plan"`, `"modify_plan"`, `"explain_exercise"` |
| `plan` | `str` | JSON string of the structured workout plan (exercises, sets, reps, rest) |
| `reasoning` | `str \| None` | Optional agent justification for the plan choices |

**Plan JSON schema (basic / injury tasks):**
```json
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
```

**Plan JSON schema (periodized program task):**
```json
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
```

## Observation Space

Each step returns a `FitscriptObservation`:

| Field | Type | Description |
|-------|------|-------------|
| `client_profile` | `dict` | Age, fitness level, goal, equipment, injuries, days/week |
| `feedback` | `str` | Human-readable grader feedback on the submitted plan |
| `score_breakdown` | `dict[str, float]` | Per-criterion partial scores |
| `task_id` | `str` | Active task identifier |
| `step_count` | `int` | Current step within the episode |
| `done` | `bool` | `True` when task complete or max steps reached |
| `reward` | `float` | Step reward in `[0.0, 1.0]` |

## Task Descriptions

### Task 1 — EASY: Basic Plan Generation (`basic_plan`)

**Client:** 35-year-old beginner, no injuries, 3 days/week, home, no equipment.

**Grader criteria (0.25 each):**
1. Plan contains exactly 3 workout days.
2. All exercises are bodyweight-only (no equipment required).
3. Each day has 4–8 exercises with `sets` and `reps` defined.
4. Beginner-appropriate: reps ≤ 15, no advanced movements (muscle-ups, pistol squats, etc.).

**Score formula:** `(criteria_met / 4)` → `[0.0, 1.0]`  
**Episode ends:** plan submitted OR after 3 steps.

---

### Task 2 — MEDIUM: Injury-Safe Plan Modification (`injury_safe_modification`)

**Client:** 30-year-old intermediate, lower-back injury, pre-generated plan contains back squats, deadlifts, and bent-over rows.

**Grader criteria (0.25 each):**
1. Deadlifts removed or replaced (Romanian deadlift / leg press / hip thrust).
2. Back squats replaced (goblet squat / wall sit / leg press).
3. Bent-over rows replaced (seated cable row / machine row).
4. Plan retains same muscle-group targets despite modifications.

**Score formula:** `(criteria_met / 4)` → `[0.0, 1.0]`  
**Episode ends:** modification submitted OR after 5 steps.

---

### Task 3 — HARD: Periodized 4-Week Program (`periodized_program`)

**Client:** 27-year-old advanced powerlifter, 5 days/week, full gym, competition in 5 weeks, weak points: upper back and lockout strength.

**Grader criteria (0.2 each):**
1. 4 distinct weeks, each with 5 training days.
2. Weeks 1–3 show progressive overload (increasing intensity/RPE).
3. Week 4 is a deload: volume reduced ≥ 40% vs week 3.
4. Competition lifts (squat, bench, deadlift) present as primary movements.
5. Bonus: accessory work targets weak points (upper back, lockout).

**Score formula:** `min(1.0, criteria_met * 0.2)` → `[0.0, 1.0]`  
**Episode ends:** full 4-week program submitted OR after 8 steps.

---

## Reward Design

- **Per-step reward:** `max(0.0, partial_score − safety_penalty)`
- **Safety penalty:** −0.3 if contraindicated exercises are present for an injured client.
- **Empty plan:** reward = 0.0.
- **Duplicate plan:** reward = 0.0 (no improvement penalty).
- All rewards are clamped to `[0.0, 1.0]`.

## Setup Instructions

### Build the Docker image
```bash
docker build -t FitScript-env:latest -f server/Dockerfile .
```

### Run locally
```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Run inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=<your_key>
export FITSCRIPT_TASK=basic_plan   # or injury_safe_modification / periodized_program

python inference.py
```

### Deploy to Hugging Face Spaces
```bash
# From the directory containing openenv.yaml
openenv push

# With options
openenv push --repo-id my-org/fitscript-env --private
```

The deployed space exposes:
- **Web Interface** at `/web`
- **API Docs** at `/docs`
- **Health Check** at `/health`
- **WebSocket** at `/ws`

### Pre-submission validation
```bash
bash validate.sh <HF_SPACE_URL> <REPO_DIR>
# Step 1: POST /reset returns HTTP 200
# Step 2: docker build succeeds
# Step 3: openenv validate passes
```

## Baseline Scores

> Run `python inference.py` for each task and record the `[END] score=...` line.

| Task | Difficulty | Baseline Score | Model |
|------|-----------|---------------|-------|
| `basic_plan` | Easy | _TBD_ | _fill before submission_ |
| `injury_safe_modification` | Medium | _TBD_ | _fill before submission_ |
| `periodized_program` | Hard | _TBD_ | _fill before submission_ |

## Project Structure

```
FitScript/
├── inference.py               # ← Hackathon entry point (REQUIRED)
├── openenv.yaml               # OpenEnv manifest with tasks section
├── pyproject.toml             # Project metadata and dependencies
├── __init__.py                # Module exports
├── client.py                  # FitscriptEnv client
├── models.py                  # FitscriptAction and FitscriptObservation
└── server/
    ├── __init__.py            # Server module exports
    ├── FitScript_environment.py   # Core environment + 3 task graders
    ├── app.py                 # FastAPI application (HTTP + WebSocket)
    └── Dockerfile             # Multi-stage container definition
```
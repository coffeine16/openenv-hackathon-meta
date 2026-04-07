# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FitScript Environment Implementation.

Simulates a real-world fitness prescription task: generating, evaluating,
and refining personalized workout plans. Supports three tasks of increasing
difficulty with deterministic graders.
"""

import json
from uuid import uuid4
from typing import Dict, Any, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FitscriptAction, FitscriptObservation
except ImportError:
    from models import FitscriptAction, FitscriptObservation


# ---------------------------------------------------------------------------
# Grader base class
# ---------------------------------------------------------------------------

class BaseTask:
    """Base class for all FitScript tasks."""

    client_profile: dict = {}
    max_steps: int = 5

    def grade(
        self, action: FitscriptAction, step: int
    ) -> Tuple[float, Dict[str, float], str]:
        """
        Returns (score: float in [0,1], breakdown: dict, feedback: str).
        Must be implemented by every concrete task.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1 - EASY: Basic Plan Generation
# ---------------------------------------------------------------------------

class BasicPlanTask(BaseTask):
    """
    Scenario: 35-year-old beginner, no injuries, 3 days/week, home, no equipment.
    Grader: 4 criteria worth 0.25 each.
    Episode ends when plan submitted OR after 3 steps.
    """

    client_profile = {
        "age": 35,
        "fitness_level": "beginner",
        "goal": "general fitness",
        "equipment": [],
        "injuries": [],
        "days_per_week": 3,
    }
    max_steps = 3

    # Exercises that require equipment --- flag any appearance
    EQUIPMENT_EXERCISES = {
        "barbell", "dumbbell", "kettlebell", "cable", "machine",
        "bench press", "squat rack", "pull-up bar", "resistance band",
        "treadmill", "stationary bike",
    }

    # Advanced movements not appropriate for beginners
    ADVANCED_MOVEMENTS = {
        "muscle-up", "muscle up", "handstand push-up", "handstand pushup",
        "pistol squat", "one-arm push-up", "planche", "front lever",
        "back lever", "dragon flag",
    }

    def grade(self, action: FitscriptAction, step: int) -> Tuple[float, Dict[str, float], str]:
        scores: Dict[str, float] = {}
        feedback_parts = []

        try:
            plan = json.loads(action.plan) if action.plan else {}
        except json.JSONDecodeError:
            plan = {}

        # Criterion 1: Plan contains exactly 3 workout days
        days = plan.get("days", plan.get("workout_days", []))
        if isinstance(days, list) and len(days) == 3:
            scores["three_days"] = 0.25
            feedback_parts.append("✓ Plan has exactly 3 workout days.")
        else:
            scores["three_days"] = 0.0
            found = len(days) if isinstance(days, list) else "unknown"
            feedback_parts.append(f"✗ Expected 3 workout days, found {found}.")

        # Criterion 2: All exercises are bodyweight-only
        all_exercises = _extract_exercises(plan)
        plan_text_lower = action.plan.lower()
        equipment_found = [e for e in self.EQUIPMENT_EXERCISES if e in plan_text_lower]
        if not equipment_found:
            scores["bodyweight_only"] = 0.25
            feedback_parts.append("✓ No equipment required --- all bodyweight exercises.")
        else:
            scores["bodyweight_only"] = 0.0
            feedback_parts.append(f"✗ Equipment-dependent exercises found: {equipment_found[:3]}.")

        # Criterion 3: Each day has 4-8 exercises with sets and reps defined
        if isinstance(days, list) and len(days) > 0:
            days_ok = 0
            for day in days:
                exs = day.get("exercises", [])
                if 4 <= len(exs) <= 8 and all(
                    e.get("sets") and e.get("reps") for e in exs
                ):
                    days_ok += 1
            if days_ok == len(days) and len(days) > 0:
                scores["exercise_structure"] = 0.25
                feedback_parts.append("✓ Each day has 4-8 exercises with sets and reps defined.")
            else:
                scores["exercise_structure"] = 0.0
                feedback_parts.append(
                    f"✗ {days_ok}/{len(days)} days have 4-8 exercises with sets+reps. "
                    "Ensure every exercise has 'sets' and 'reps' fields."
                )
        else:
            scores["exercise_structure"] = 0.0
            feedback_parts.append("✗ Cannot evaluate exercise structure: no days found.")

        # Criterion 4: Beginner-appropriate (reps <= 15, no advanced movements)
        advanced_found = [m for m in self.ADVANCED_MOVEMENTS if m in plan_text_lower]
        reps_too_high = _check_reps_exceed(plan, max_reps=15)
        if not advanced_found and not reps_too_high:
            scores["beginner_appropriate"] = 0.25
            feedback_parts.append("✓ Plan is beginner-appropriate (no advanced movements, reps ≤ 15).")
        else:
            scores["beginner_appropriate"] = 0.0
            if advanced_found:
                feedback_parts.append(f"✗ Advanced movements not suitable for beginners: {advanced_found}.")
            if reps_too_high:
                feedback_parts.append("✗ Some exercises have reps > 15 --- too high for a beginner.")

        score = sum(scores.values())
        feedback = " ".join(feedback_parts)
        return score, scores, feedback


# ---------------------------------------------------------------------------
# Task 2 - MEDIUM: Injury-Safe Plan Modification
# ---------------------------------------------------------------------------

class InjurySafeTask(BaseTask):
    """
    Scenario: Intermediate client with lower-back injury. Pre-generated plan
    contains back squats, deadlifts, and bent-over rows. Agent must modify safely.
    Episode ends when modification submitted OR after 5 steps.
    """

    client_profile = {
        "age": 30,
        "fitness_level": "intermediate",
        "goal": "strength maintenance",
        "equipment": ["barbell", "dumbbells", "cables", "machines"],
        "injuries": ["lower back"],
        "days_per_week": 4,
        "initial_plan": {
            "days": [
                {
                    "name": "Day 1 - Lower Body",
                    "exercises": [
                        {"name": "Back Squat", "sets": 4, "reps": 8},
                        {"name": "Deadlift", "sets": 3, "reps": 5},
                        {"name": "Leg Press", "sets": 3, "reps": 10},
                        {"name": "Calf Raises", "sets": 4, "reps": 15},
                    ],
                },
                {
                    "name": "Day 2 - Upper Body",
                    "exercises": [
                        {"name": "Bench Press", "sets": 4, "reps": 8},
                        {"name": "Bent-Over Row", "sets": 4, "reps": 8},
                        {"name": "Overhead Press", "sets": 3, "reps": 10},
                        {"name": "Pull-Up", "sets": 3, "reps": "max"},
                    ],
                },
            ]
        },
    }
    max_steps = 5

    DEADLIFT_REPLACEMENTS = {
        "romanian deadlift", "rdl", "leg press", "leg curl",
        "hip thrust", "glute bridge", "trap bar deadlift",
    }
    SQUAT_REPLACEMENTS = {
        "goblet squat", "wall sit", "wall squat", "leg press",
        "box squat", "safety bar squat", "hack squat",
    }
    ROW_REPLACEMENTS = {
        "seated cable row", "seated row", "machine row",
        "chest-supported row", "chest supported row",
        "t-bar row", "seal row",
    }
    ORIGINAL_MUSCLE_GROUPS = {"quads", "hamstrings", "glutes", "back", "chest", "shoulders"}

    def grade(self, action: FitscriptAction, step: int) -> Tuple[float, Dict[str, float], str]:
        scores: Dict[str, float] = {}
        feedback_parts = []
        plan_text_lower = action.plan.lower()

        # Criterion 1: Deadlifts removed or replaced with safe alternatives
        has_deadlift = "deadlift" in plan_text_lower and not any(
            r in plan_text_lower for r in self.DEADLIFT_REPLACEMENTS
        )
        raw_deadlift = "deadlift" in plan_text_lower and "romanian" not in plan_text_lower and "rdl" not in plan_text_lower
        if not raw_deadlift:
            scores["deadlift_removed"] = 0.25
            feedback_parts.append("✓ Conventional deadlift removed or replaced safely.")
        else:
            scores["deadlift_removed"] = 0.0
            feedback_parts.append(
                "✗ Conventional deadlift still present. Replace with Romanian deadlift, leg press, or hip thrust."
            )

        # Criterion 2: Back squats replaced with safe alternatives
        has_back_squat = "back squat" in plan_text_lower
        if not has_back_squat:
            scores["squat_replaced"] = 0.25
            feedback_parts.append("✓ Back squat removed or replaced safely.")
        else:
            scores["squat_replaced"] = 0.0
            feedback_parts.append(
                "✗ Back squat still present. Replace with goblet squat, wall sit, or leg press."
            )

        # Criterion 3: Bent-over rows replaced with seated/machine variants
        has_bent_over_row = "bent-over row" in plan_text_lower or "bent over row" in plan_text_lower
        if not has_bent_over_row:
            scores["rows_replaced"] = 0.25
            feedback_parts.append("✓ Bent-over rows removed or replaced with spine-neutral variant.")
        else:
            scores["rows_replaced"] = 0.0
            feedback_parts.append(
                "✗ Bent-over rows still present. Replace with seated cable rows or machine rows."
            )

        # Criterion 4: Plan retains same muscle group targets
        # Proxy: check that back/leg work still appears in the plan
        back_work = any(
            t in plan_text_lower
            for t in ["row", "pull", "lat", "back", "rhomboid"]
        )
        leg_work = any(
            t in plan_text_lower
            for t in ["squat", "press", "lunge", "hip", "glute", "quad", "hamstring", "leg"]
        )
        if back_work and leg_work:
            scores["muscle_targets_retained"] = 0.25
            feedback_parts.append("✓ Original muscle groups (back, legs) still targeted despite modifications.")
        else:
            scores["muscle_targets_retained"] = 0.0
            missing = []
            if not back_work:
                missing.append("back")
            if not leg_work:
                missing.append("legs")
            feedback_parts.append(
                f"✗ Missing muscle group coverage: {missing}. Ensure modifications keep the same target areas."
            )

        score = sum(scores.values())
        feedback = " ".join(feedback_parts)
        return score, scores, feedback


# ---------------------------------------------------------------------------
# Task 3 - HARD: Periodized 4-Week Program
# ---------------------------------------------------------------------------

class PeriodizedProgramTask(BaseTask):
    """
    Scenario: Advanced powerlifter, 5 days/week, full gym, competition in 5 weeks.
    Needs 4-week block with deload in week 4.
    Episode ends when full program submitted OR after 8 steps.
    """

    client_profile = {
        "age": 27,
        "fitness_level": "advanced",
        "goal": "powerlifting competition prep",
        "equipment": ["full gym", "barbell", "squat rack", "bench", "deadlift platform"],
        "injuries": [],
        "days_per_week": 5,
        "competition_weeks_out": 5,
        "weak_points": ["upper back", "lockout strength"],
        "current_maxes": {"squat": 180, "bench": 120, "deadlift": 220},
    }
    max_steps = 8

    COMPETITION_LIFTS = {"squat", "bench", "bench press", "deadlift"}

    def grade(self, action: FitscriptAction, step: int) -> Tuple[float, Dict[str, float], str]:
        scores: Dict[str, float] = {}
        feedback_parts = []

        try:
            plan = json.loads(action.plan) if action.plan else {}
        except json.JSONDecodeError:
            plan = {}

        weeks = plan.get("weeks", [])

        # Criterion 1: 4 distinct weeks, each with 5 training days
        if isinstance(weeks, list) and len(weeks) == 4:
            all_five_days = all(
                len(w.get("days", w.get("training_days", []))) == 5
                for w in weeks
            )
            if all_five_days:
                scores["week_structure"] = 0.2
                feedback_parts.append("✓ 4 weeks present, each with 5 training days.")
            else:
                scores["week_structure"] = 0.1
                feedback_parts.append(
                    "~ 4 weeks present but not all weeks have exactly 5 training days."
                )
        else:
            scores["week_structure"] = 0.0
            found_weeks = len(weeks) if isinstance(weeks, list) else "unknown"
            feedback_parts.append(
                f"✗ Expected 4 weeks with 5 days each. Found {found_weeks} weeks."
            )

        # Criterion 2: Weeks 1-3 show progressive overload
        if isinstance(weeks, list) and len(weeks) >= 3:
            intensities = []
            for w in weeks[:3]:
                # Accept intensity as explicit field or infer from RPE/percentage keywords
                intensity = w.get("intensity") or w.get("avg_rpe") or w.get("percentage")
                if intensity is None:
                    # Try to infer from week label/description
                    desc = str(w).lower()
                    if "heavy" in desc or "high" in desc:
                        intensity = 85
                    elif "moderate" in desc or "medium" in desc:
                        intensity = 75
                    else:
                        intensity = None
                intensities.append(intensity)

            if all(i is not None for i in intensities) and intensities[0] < intensities[1] < intensities[2]:
                scores["progressive_overload"] = 0.2
                feedback_parts.append("✓ Weeks 1-3 show clear progressive overload (increasing intensity).")
            elif all(i is not None for i in intensities):
                scores["progressive_overload"] = 0.1
                feedback_parts.append(
                    "~ Intensity values present but progressive overload pattern not clearly ascending across weeks 1-3."
                )
            else:
                scores["progressive_overload"] = 0.0
                feedback_parts.append(
                    "✗ Cannot verify progressive overload. Add 'intensity', 'avg_rpe', or 'percentage' fields to each week."
                )
        else:
            scores["progressive_overload"] = 0.0
            feedback_parts.append("✗ Fewer than 3 weeks present; cannot verify progressive overload.")

        # Criterion 3: Week 4 is a deload (volume reduced >= 40% vs week 3)
        if isinstance(weeks, list) and len(weeks) == 4:
            w3 = weeks[2]
            w4 = weeks[3]
            w3_vol = _estimate_volume(w3)
            w4_vol = _estimate_volume(w4)
            is_deload_label = "deload" in str(w4).lower()
            if w3_vol > 0 and w4_vol > 0:
                reduction = (w3_vol - w4_vol) / w3_vol
                if reduction >= 0.40:
                    scores["deload_week"] = 0.2
                    feedback_parts.append(
                        f"✓ Week 4 deload: volume reduced by {reduction*100:.0f}% vs week 3."
                    )
                elif is_deload_label:
                    scores["deload_week"] = 0.1
                    feedback_parts.append(
                        "~ Week 4 labeled as deload but volume reduction < 40%. Reduce total sets/volume further."
                    )
                else:
                    scores["deload_week"] = 0.0
                    feedback_parts.append(
                        f"✗ Week 4 volume only reduced by {reduction*100:.0f}%. Deload requires >= 40% reduction."
                    )
            elif is_deload_label:
                scores["deload_week"] = 0.1
                feedback_parts.append(
                    "~ Week 4 labeled as deload but no volume data to verify the 40% reduction threshold."
                )
            else:
                scores["deload_week"] = 0.0
                feedback_parts.append(
                    "✗ Week 4 not identified as a deload and volume data insufficient to verify."
                )
        else:
            scores["deload_week"] = 0.0
            feedback_parts.append("✗ Fewer than 4 weeks present; cannot evaluate deload week.")

        # Criterion 4: Competition lifts appear as primary movements on separate days
        plan_text_lower = action.plan.lower()
        squat_present = "squat" in plan_text_lower
        bench_present = "bench" in plan_text_lower
        deadlift_present = "deadlift" in plan_text_lower
        if squat_present and bench_present and deadlift_present:
            scores["competition_lifts"] = 0.2
            feedback_parts.append("✓ All three competition lifts (squat, bench, deadlift) present as primary movements.")
        else:
            missing = []
            if not squat_present:
                missing.append("squat")
            if not bench_present:
                missing.append("bench press")
            if not deadlift_present:
                missing.append("deadlift")
            scores["competition_lifts"] = 0.0
            feedback_parts.append(f"✗ Missing competition lifts: {missing}.")

        # Criterion 5 (bonus): Accessory work targets weak points (upper back, lockout)
        weak_point_keywords = ["face pull", "upper back", "row", "rdl", "pause", "lockout", "band pull apart", "rear delt"]
        accessory_bonus = sum(1 for kw in weak_point_keywords if kw in plan_text_lower)
        if accessory_bonus >= 3:
            scores["accessory_weak_points"] = 0.2
            feedback_parts.append("✓ Accessory work targets weak points (upper back, lockout strength).")
        elif accessory_bonus >= 1:
            scores["accessory_weak_points"] = 0.1
            feedback_parts.append("~ Some accessory work present but weak points (upper back, lockout) not fully addressed.")
        else:
            scores["accessory_weak_points"] = 0.0
            feedback_parts.append("✗ No accessory work targeting weak points (upper back, lockout strength).")

        score = min(1.0, sum(scores.values()))
        feedback = " ".join(feedback_parts)
        return score, scores, feedback


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _extract_exercises(plan: dict) -> list:
    """Flatten all exercises from all days in a plan."""
    exercises = []
    for day in plan.get("days", plan.get("workout_days", [])):
        if isinstance(day, dict):
            exercises.extend(day.get("exercises", []))
    return exercises


def _check_reps_exceed(plan: dict, max_reps: int) -> bool:
    """Return True if any exercise in the plan has reps > max_reps."""
    for ex in _extract_exercises(plan):
        reps = ex.get("reps")
        if isinstance(reps, (int, float)) and reps > max_reps:
            return True
    return False


def _estimate_volume(week: dict) -> float:
    """Estimate total volume (sets × reps) across all days in a week."""
    total = 0
    for day in week.get("days", week.get("training_days", [])):
        if isinstance(day, dict):
            for ex in day.get("exercises", []):
                sets = ex.get("sets", 0)
                reps = ex.get("reps", 0)
                if isinstance(sets, (int, float)) and isinstance(reps, (int, float)):
                    total += sets * reps
    # Also accept a flat 'total_sets' key on the week
    if total == 0:
        total = week.get("total_sets", 0) * 8  # assume ~8 reps avg if only sets given
    return float(total)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, BaseTask] = {
    "basic_plan": BasicPlanTask(),
    "injury_safe_modification": InjurySafeTask(),
    "periodized_program": PeriodizedProgramTask(),
}


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class FitscriptEnvironment(Environment):
    """
    FitScript fitness prescription environment.

    Three tasks of increasing difficulty:
      - basic_plan (easy): generate a 3-day bodyweight beginner plan
      - injury_safe_modification (medium): modify a plan for a lower-back-injured client
      - periodized_program (hard): design a 4-week periodized powerlifting block

    Rewards are always in [0.0, 1.0]. Episodes terminate on task completion
    (score >= 0.99) or when max_steps is reached.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "basic_plan"):
        """
        Initialize the FitScript environment.

        Args:
            task_id: One of 'basic_plan', 'injury_safe_modification', 'periodized_program'.
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}"
            )
        self._task_id = task_id
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_plan: str = ""

    def reset(self) -> FitscriptObservation:
        """
        Reset the environment for the current task.

        Returns:
            FitscriptObservation with the client profile and welcome message.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._last_plan = ""

        task = TASKS[self._task_id]

        return FitscriptObservation(
            client_profile=task.client_profile,
            feedback="Welcome! Review the client profile and generate a plan.",
            score_breakdown={},
            task_id=self._task_id,
            step_count=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: FitscriptAction) -> FitscriptObservation:  # type: ignore[override]
        """
        Execute a step: grade the submitted plan and return feedback.

        Args:
            action: FitscriptAction with action_type, plan JSON string, and optional reasoning.

        Returns:
            FitscriptObservation with score breakdown and feedback.
        """
        self._state.step_count += 1
        task = TASKS[self._task_id]

        # Penalty: empty or null plan
        if not action.plan or action.plan.strip() in ("", "null", "{}"):
            return FitscriptObservation(
                client_profile=task.client_profile,
                feedback="✗ Empty or null plan submitted. Please provide a structured workout plan.",
                score_breakdown={},
                task_id=self._task_id,
                step_count=self._state.step_count,
                done=self._state.step_count >= task.max_steps,
                reward=0.0,
            )

        # Penalty: identical plan submitted twice in a row
        if action.plan == self._last_plan:
            return FitscriptObservation(
                client_profile=task.client_profile,
                feedback="✗ Identical plan submitted twice. Please revise based on the previous feedback.",
                score_breakdown={},
                task_id=self._task_id,
                step_count=self._state.step_count,
                done=self._state.step_count >= task.max_steps,
                reward=0.0,
            )

        self._last_plan = action.plan

        # Grade the plan
        score, breakdown, feedback = task.grade(action, self._state.step_count)

        # Safety penalty: contraindicated exercises for injured clients
        injuries = task.client_profile.get("injuries", [])
        if injuries:
            plan_lower = action.plan.lower()
            CONTRAINDICATED = {
                "lower back": ["deadlift", "back squat", "good morning", "bent-over row"],
                "knee": ["lunge", "leg press", "deep squat", "box jump"],
                "shoulder": ["overhead press", "upright row", "behind neck"],
            }
            for injury in injuries:
                banned = CONTRAINDICATED.get(injury, [])
                if any(b in plan_lower for b in banned):
                    score = max(0.0, score - 0.3)
                    feedback += " ⚠️ Safety penalty applied: plan contains exercises contraindicated for the client's injury."
                    break

        # Clamp to [0.0, 1.0]
        score = max(0.0, min(1.0, score))

        done = score >= 0.99 or self._state.step_count >= task.max_steps

        return FitscriptObservation(
            client_profile=task.client_profile,
            feedback=feedback,
            score_breakdown=breakdown,
            task_id=self._task_id,
            step_count=self._state.step_count,
            done=done,
            reward=score,
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
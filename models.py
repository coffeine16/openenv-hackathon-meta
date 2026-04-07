# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the FitScript Environment.

FitScript simulates a real-world AI fitness prescription task:
generating, evaluating, and refining personalized workout plans.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Optional, Dict, Any


class FitscriptAction(Action):
    """Action for the FitScript environment --- fitness plan generation/modification."""

    action_type: str = Field(
        ...,
        description="One of: 'generate_plan' | 'modify_plan' | 'explain_exercise'"
    )
    plan: str = Field(
        default="",
        description="JSON string of structured workout plan (exercises, sets, reps, rest)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent justification for the plan choices"
    )


class FitscriptObservation(Observation):
    """Observation from the FitScript environment --- client profile and plan feedback."""

    client_profile: Dict[str, Any] = Field(
        default_factory=dict,
        description="Client info: age, fitness_level, goal, equipment, injuries, days_per_week"
    )
    feedback: str = Field(
        default="",
        description="Environment feedback on the last submitted plan"
    )
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Partial scores per criterion (safety, completeness, progression)"
    )
    task_id: str = Field(
        default="",
        description="Current task identifier"
    )
    step_count: int = Field(
        default=0,
        description="Current step within the episode"
    )
    # done and reward are inherited from the Observation base class
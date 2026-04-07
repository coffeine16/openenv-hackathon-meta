# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FitScript Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FitscriptAction, FitscriptObservation


class FitscriptEnv(
    EnvClient[FitscriptAction, FitscriptObservation, State]
):
    """
    Client for the FitScript Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with FitscriptEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.client_profile)
        ...
        ...     result = client.step(FitscriptAction(
        ...         action_type="generate_plan",
        ...         plan='{"days": [...]}',
        ...         reasoning="Beginner-safe bodyweight plan"
        ...     ))
        ...     print(result.observation.feedback)
        ...     print(result.reward)

    Example with Docker:
        >>> client = FitscriptEnv.from_docker_image("FitScript-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(FitscriptAction(
        ...         action_type="generate_plan",
        ...         plan='{"days": [...]}'
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: FitscriptAction) -> Dict:
        """
        Convert FitscriptAction to JSON payload for step message.

        Args:
            action: FitscriptAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {
            "action_type": action.action_type,
            "plan": action.plan,
        }
        if action.reasoning is not None:
            payload["reasoning"] = action.reasoning
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[FitscriptObservation]:
        """
        Parse server response into StepResult[FitscriptObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with FitscriptObservation
        """
        obs_data = payload.get("observation", {})
        observation = FitscriptObservation(
            client_profile=obs_data.get("client_profile", {}),
            feedback=obs_data.get("feedback", ""),
            score_breakdown=obs_data.get("score_breakdown", {}),
            task_id=obs_data.get("task_id", ""),
            step_count=obs_data.get("step_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
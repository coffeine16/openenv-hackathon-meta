# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the FitScript Environment.

The task is selected via the FITSCRIPT_TASK environment variable (default: basic_plan).
Valid values: basic_plan | injury_safe_modification | periodized_program

Endpoints:
    POST /reset  — Reset the environment
    POST /step   — Execute an action
    GET  /state  — Get current environment state
    GET  /schema — Get action/observation schemas
    WS   /ws     — WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    FITSCRIPT_TASK=basic_plan uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

import os
import functools

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install with: pip install openenv-core"
    ) from e

try:
    from models import FitscriptAction, FitscriptObservation
    from server.FitScript_environment import FitscriptEnvironment
except ModuleNotFoundError:
    from ..models import FitscriptAction, FitscriptObservation
    from .FitScript_environment import FitscriptEnvironment

# Read the task from the environment variable; default to basic_plan
FITSCRIPT_TASK = os.environ.get("FITSCRIPT_TASK", "basic_plan")

VALID_TASKS = {"basic_plan", "injury_safe_modification", "periodized_program"}
if FITSCRIPT_TASK not in VALID_TASKS:
    raise ValueError(
        f"Invalid FITSCRIPT_TASK='{FITSCRIPT_TASK}'. "
        f"Must be one of: {sorted(VALID_TASKS)}"
    )

# Use functools.partial so create_app can instantiate the env with the right task_id
EnvFactory = functools.partial(FitscriptEnvironment, task_id=FITSCRIPT_TASK)

# Create the FastAPI app
app = create_app(
    EnvFactory,
    FitscriptAction,
    FitscriptObservation,
    env_name="FitScript",
    max_concurrent_envs=4,
)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
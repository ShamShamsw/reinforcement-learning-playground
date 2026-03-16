"""
main.py - Entry point for Reinforcement Learning Playground
===========================================================

Thin-controller module that only orchestrates calls to:
    - operations.py for workflow execution
    - display.py for presentation formatting
"""

from display import format_header, format_run_report, format_startup_guide
from models import create_project_config
from operations import load_environment_settings, run_core_flow


def main() -> None:
    """Run one complete RL training and evaluation session.

    Returns:
        None
    """
    config = create_project_config()
    env_settings = load_environment_settings()
    print(format_header())
    print(format_startup_guide(config, env_settings))
    run_summary = run_core_flow(config=config)
    print(format_run_report(run_summary))


if __name__ == "__main__":
    # Guard keeps imports side-effect free for tests and reuse.
    main()

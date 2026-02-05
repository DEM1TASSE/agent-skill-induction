#!/usr/bin/env python3
"""
WebArena-Verified BrowserGym Runner

Same structure as run_demo.py but uses VerifiedWebArenaTask which integrates
WebArena-Verified evaluation into the task's validate() method.

Usage:
    python run_demo_verified.py --task_name webarena_verified.401
    python run_demo_verified.py --task_name webarena_verified.401 --headless
"""

import os
import sys
from pathlib import Path

# Use local browsergym (with record_har support) instead of installed package
_ASI_ROOT = Path(__file__).parent
_BROWSERGYM_SRC = _ASI_ROOT.parent / "browsergym" / "experiments" / "src"
if _BROWSERGYM_SRC.exists():
    sys.path.insert(0, str(_BROWSERGYM_SRC))

import argparse
import ast
import json
import re
import shutil
from pathlib import Path

from agent import DemoAgentArgs
from patch_with_custom_exec import patch_with_custom_exec
from task_verified import register_verified_tasks

from browsergym.experiments import EnvArgs, ExpArgs, get_exp_result


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run WebArena-Verified task.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="litellm/neulab/claude-3-5-sonnet-20241022",
        choices=[
            "litellm/neulab/claude-3-5-sonnet-20241022",
            "litellm/neulab/gpt-4o-2024-05-13",
            "litellm/neulab/claude-sonnet-4-20250514",
            "litellm/neulab/claude-sonnet-4-5-20250929",
            "litellm_proxy/neulab/claude-sonnet-4-20250514",
            "azure/gpt-4o",
            "gpt-4o",
        ],
        help="LLM model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Task name, e.g., 'webarena_verified.401'",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in observation.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in observation.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in observation.",
    )
    parser.add_argument(
        "--websites",
        type=str,
        nargs="+",
        default=[],
        choices=["shopping", "admin", "reddit", "gitlab", "map"],
        help="Website(s) for action space.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Maximum steps.",
    )
    parser.add_argument(
        "--action_path",
        type=str,
        default=None,
        help="Path to predefined actions.",
    )
    parser.add_argument(
        "--memory_path",
        type=str,
        default=None,
        help="Path to workflow memory.",
    )
    parser.add_argument(
        "--results",
        default="results/verified",
        help="Directory where task outputs are written (default: results/verified).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless.",
    )

    return parser.parse_args()


def main():
    print("""\
--- WebArena-Verified Agent ---
Uses VerifiedWebArenaTask with integrated evaluation.
""")

    # Register webarena_verified tasks
    register_verified_tasks()

    args = parse_args()
    os.environ["BROWSERGYM_RESULTS_ROOT"] = args.results

    # Extract task_id from task_name (e.g., "webarena_verified.401" -> 401)
    task_id = args.task_name.split(".")[-1]

    # Results directory structure: results/verified/{task_id}/
    results_root = args.results
    task_output_dir = f"{results_root}/{task_id}"
    os.makedirs(task_output_dir, exist_ok=True)

    print(f"Task: {args.task_name}")
    print(f"Results: {task_output_dir}")

    # Load predefined actions if provided
    if args.action_path is not None and os.path.exists(args.action_path):
        actions = open(args.action_path, "r").read()
        if actions.strip():
            actions = actions.splitlines()
        else:
            actions = []
    else:
        actions = []

    # Agent config (same as run_demo.py)
    agent_args = DemoAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
        websites=args.websites,
        actions=actions,
        memory=args.memory_path,
    )

    patch_with_custom_exec(agent_args)

    # Environment config - enable HAR recording for network trace evaluation
    env_kwargs = {
        "task_name": args.task_name,
        "task_seed": None,
        "max_steps": args.max_steps,
        "headless": args.headless,
        "record_har": True,
    }
    env_args = EnvArgs(**env_kwargs)

    # Experiment config
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # Run - use task output dir
    exp_args.prepare(task_output_dir)
    os.environ["BROWSERGYM_EXP_DIR"] = str(exp_args.exp_dir)
    env_args.task_kwargs = {"task_id": int(task_id), "exp_dir": str(exp_args.exp_dir)}
    exp_args.run()

    # Post-hoc WebArena-Verified evaluation (HAR is available after context close)
    exp_dir = Path(exp_args.exp_dir)
    agent_response_path = exp_dir / "agent_response.json"
    agent_response_ok = agent_response_path.exists()
    if not agent_response_path.exists():
        log_path = exp_dir / "experiment.log"
        extracted = None
        if log_path.exists():
            text = log_path.read_text()
            matches = re.findall(
                r"save_agent_response\s*\(\s*(\{[\s\S]*?\})\s*\)", text
            )
            if matches:
                raw = matches[-1]
                try:
                    extracted = json.loads(raw.replace("None", "null"))
                except json.JSONDecodeError:
                    try:
                        extracted = ast.literal_eval(raw)
                    except (SyntaxError, ValueError):
                        extracted = None

        agent_response_ok = False
        if extracted is not None:
            with agent_response_path.open("w") as f:
                json.dump(extracted, f, indent=2)
            agent_response_ok = True
        else:
            cwd_response = Path("agent_response.json")
            if cwd_response.exists():
                shutil.copy2(cwd_response, agent_response_path)
                agent_response_ok = True
            else:
                fallback = {
                    "task_type": "RETRIEVE",
                    "status": "UNKNOWN_ERROR",
                    "retrieved_data": None,
                    "error_details": "No save_agent_response found in experiment.log",
                }
                with agent_response_path.open("w") as f:
                    json.dump(fallback, f, indent=2)
                agent_response_ok = False

    har_path = exp_dir / "network.har"
    if not har_path.exists():
        error_result = {
            "task_id": int(task_id),
            "score": 0.0,
            "status": "error",
            "error": "HAR file missing; skipping evaluation",
        }
        with (exp_dir / "eval_result.json").open("w") as f:
            json.dump(error_result, f, indent=2)
        result = error_result
        print(f"Warning: HAR file not found at {har_path}; evaluation skipped")
    elif not agent_response_ok:
        error_result = {
            "task_id": int(task_id),
            "score": 0.0,
            "status": "error",
            "error": "Agent response missing; skipping evaluation",
        }
        with (exp_dir / "eval_result.json").open("w") as f:
            json.dump(error_result, f, indent=2)
        result = error_result
        print("Warning: agent_response.json missing; evaluation skipped")
    else:
        from webarena_verified_utils import run_verified_evaluation

        result = run_verified_evaluation(
            task_id=int(task_id),
            exp_dir=exp_dir,
        )

    # Print results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")

    print(f"\nEvaluation Result:")
    print(f"  Task ID: {result.get('task_id', task_id)}")
    print(f"  Score: {result.get('score', 'N/A')}")
    print(f"  Status: {result.get('status', 'N/A')}")
    if result.get("evaluators_results"):
        print("  Evaluators:")
        for er in result["evaluators_results"]:
            print(
                f"    - {er['evaluator_name']}: {er['status']} (score: {er['score']})"
            )

    print(f"\nResults saved to: {exp_args.exp_dir}")


if __name__ == "__main__":
    main()

import os
import json
import argparse
from pathlib import Path

# locally defined agent
from agent import DemoAgentArgs
from patch_with_custom_exec import patch_with_custom_exec

# browsergym experiments utils
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
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="litellm/neulab/claude-3-5-sonnet-20241022",
        choices=[
            "litellm/neulab/claude-3-5-sonnet-20241022",
            "litellm/neulab/gpt-4o-2024-05-13",
            "litellm/neulab/claude-sonnet-4-20250514",
            "litellm/neulab/claude-sonnet-4-5-20250929",
            "azure/gpt-4o",
            "azure/gpt-4o-mini",
            "gpt-4o",
        ],
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    parser.add_argument(
        "--websites", type=str, nargs='+', default=[],
        choices=["shopping", "admin", "reddit", "gitlab", "map"],
        help="Name of the website(s) to run the agent on. Used to define agent's action space.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=10,
        help="Maximum number of steps to run the agent.",
    )
    
    # debug
    parser.add_argument(
        "--action_path", type=str, default=None, # "debug_actions/test.txt",
        help="Path to the specified actions for agents to take.",
    )
    parser.add_argument(
        "--memory_path", type=str, default=None, # "memory/test.txt",
        help="Path to the workflow memory.",
    )
    parser.add_argument(
        "--rename_to", type=str, default=None,
        help="If specified, rename the experiment folder to the specified name.",
    )
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode.")
    
    # HAR recording and WebArena-Verified evaluation
    parser.add_argument(
        "--record-har",
        action="store_true",
        help="Enable HAR (HTTP Archive) recording for network trace evaluation.",
    )
    parser.add_argument(
        "--eval-verified",
        action="store_true",
        help="Run WebArena-Verified evaluation after experiment completion.",
    )
    parser.add_argument(
        "--verified-config",
        type=str,
        default=None,
        help="Path to WebArena-Verified config file (default: ../verified_config.json).",
    )

    return parser.parse_args()


def extract_task_id(task_name: str) -> int | None:
    """Extract task_id from task_name like 'webarena.117'."""
    if task_name.startswith("webarena."):
        try:
            return int(task_name.split(".")[-1])
        except ValueError:
            return None
    return None


def main():
    print(
        """\
--- WARNING ---
This is a basic agent for demo purposes.
Visit AgentLab for more capable agents with advanced features.
https://github.com/ServiceNow/AgentLab"""
    )

    args = parse_args()
    if args.rename_to is None:
        args.rename_to = args.task_name

    if args.action_path is not None and os.path.exists(args.action_path):
        actions = open(args.action_path, 'r').read()
        if actions.strip():
            actions = actions.splitlines()
        else:
            actions = []
    else:
        actions = []
    
    # setting up agent config
    agent_args = DemoAgentArgs(
        model_name=args.model_name,
        chat_mode=False,
        demo_mode="default" if args.visual_effects else "off",
        use_html=args.use_html,
        use_axtree=args.use_axtree,
        use_screenshot=args.use_screenshot,
        websites=args.websites,
        actions=tuple(actions),
        memory=args.memory_path,
    )
    
    patch_with_custom_exec(agent_args)

    # Determine if HAR recording is needed
    record_har = args.record_har or args.eval_verified
    
    # setting up environment config
    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,  # keep the browser open
        record_har=record_har,  # Enable HAR recording if requested
        # viewport={"width": 1500, "height": 1280},  # can be played with if needed
    )

    # for openended task, set environment and agent to interactive chat mode on a start url
    if args.task_name == "openended":
        agent_args.chat_mode = True
        env_args.wait_for_user_message = False # True
        env_args.task_kwargs = {"start_url": args.start_url}

    # setting up the experiment
    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=agent_args,
    )

    # running and logging results
    exp_args.prepare("./results")
    print(f"Experiment directory: {exp_args.exp_dir}")
    
    if record_har:
        print(f"HAR recording enabled, will save to: {exp_args.exp_dir}/network.har")
    
    exp_args.run()

    # loading and printing results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")
    
    # Rename result directory if requested
    final_exp_dir = exp_args.exp_dir
    if args.rename_to is not None:
        new_path = Path(f"results/{args.rename_to}")
        if new_path.exists():
            import shutil
            # Backup existing directory
            backup_path = Path(f"results/_{args.rename_to}_backup")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            new_path.rename(backup_path)
        os.rename(exp_args.exp_dir, new_path)
        final_exp_dir = new_path
        print(f"Renamed experiment directory to: {final_exp_dir}")
    
    # WebArena-Verified evaluation
    if args.eval_verified:
        task_id = extract_task_id(args.task_name)
        if task_id is None:
            print(f"Warning: Could not extract task_id from '{args.task_name}', skipping evaluation")
        else:
            print(f"\n{'='*60}")
            print(f"Running WebArena-Verified evaluation for task {task_id}")
            print(f"{'='*60}")
            
            from webarena_verified_utils import run_verified_evaluation, trim_har_file
            
            exp_dir = Path(final_exp_dir)
            config_path = Path(args.verified_config) if args.verified_config else None
            
            # Check for HAR file
            har_path = exp_dir / "network.har"
            har_trimmed = exp_dir / "network_trimmed.har"
            
            if har_path.exists():
                print(f"HAR file found: {har_path} ({har_path.stat().st_size:,} bytes)")
                try:
                    trim_har_file(har_path, har_trimmed)
                    print(f"HAR trimmed: {har_trimmed} ({har_trimmed.stat().st_size:,} bytes)")
                except Exception as e:
                    print(f"Warning: Could not trim HAR: {e}")
            else:
                print(f"Warning: HAR file not found at {har_path}")
            
            # Check for agent_response.json (agent needs to create this)
            agent_response_path = exp_dir / "agent_response.json"
            if not agent_response_path.exists():
                print(f"Warning: agent_response.json not found, creating fallback response")
                fallback = {
                    "task_type": "RETRIEVE",
                    "status": "UNKNOWN_ERROR",
                    "retrieved_data": None,
                    "error_details": "Agent did not create agent_response.json"
                }
                with agent_response_path.open("w") as f:
                    json.dump(fallback, f, indent=2)
            
            # Run evaluation
            result = run_verified_evaluation(
                task_id=task_id,
                exp_dir=exp_dir,
                config_path=config_path,
            )
            
            print(f"\n{'='*60}")
            print(f"Evaluation Result:")
            print(f"  Task ID: {task_id}")
            print(f"  Score: {result.get('score', 'N/A')}")
            print(f"  Status: {result.get('status', 'N/A')}")
            if result.get('evaluators_results'):
                print(f"  Evaluators:")
                for er in result['evaluators_results']:
                    print(f"    - {er['evaluator_name']}: {er['status']} (score: {er['score']})")
            print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

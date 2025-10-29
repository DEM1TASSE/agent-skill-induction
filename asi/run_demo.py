import os
import argparse

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
        choices=["litellm/neulab/claude-3-5-sonnet-20241022", "litellm/neulab/gpt-4o-2024-05-13", "litellm/neulab/claude-sonnet-4-20250514", "gpt-4o"],
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

    return parser.parse_args()


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

    # setting up environment config
    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,  # keep the browser open
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
    exp_args.run()

    # loading and printing results
    exp_result = get_exp_result(exp_args.exp_dir)
    exp_record = exp_result.get_exp_record()

    for key, val in exp_record.items():
        print(f"{key}: {val}")
    
    if args.rename_to is not None:
        os.rename(exp_args.exp_dir, f"results/{args.rename_to}")


if __name__ == "__main__":
    main()

"""WebArena-Verified utilities for BrowserGym integration.

This module provides:
- Task loading from webarena-verified.json dataset
- URL resolution for placeholder URLs (__SHOPPING__, etc.)
- HAR file trimming utilities
- Evaluation wrapper

Usage:
    from webarena_verified_utils import (
        load_verified_task,
        resolve_url,
        run_verified_evaluation,
    )
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Paths relative to the asi directory
_ASI_ROOT = Path(__file__).parent
_PROJECT_ROOT = _ASI_ROOT.parent
_WEBARENA_VERIFIED_DIR = _PROJECT_ROOT / "webarena-verified"
_DATASET_PATH = _WEBARENA_VERIFIED_DIR / "assets" / "dataset" / "webarena-verified.json"
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "verified_config.json"


def load_verified_task(task_name: str) -> dict[str, Any]:
    """Load task from webarena-verified dataset.
    
    Args:
        task_name: Format "webarena.{task_id}" or just the task_id as string/int
        
    Returns:
        Task data dictionary from webarena-verified.json
        
    Raises:
        ValueError: If task not found
    """
    # Parse task_id from task_name
    if isinstance(task_name, int):
        task_id = task_name
    elif task_name.startswith("webarena_verified."):
        task_id = int(task_name.split(".")[-1])
    elif task_name.startswith("webarena."):
        task_id = int(task_name.split(".")[-1])
    else:
        try:
            task_id = int(task_name)
        except ValueError:
            raise ValueError(f"Cannot parse task_id from task_name: {task_name}")
    
    if not _DATASET_PATH.exists():
        raise FileNotFoundError(f"WebArena-Verified dataset not found at {_DATASET_PATH}")
    
    with _DATASET_PATH.open() as f:
        tasks = json.load(f)
    
    for task in tasks:
        if task["task_id"] == task_id:
            return task
    
    raise ValueError(f"Task {task_id} not found in webarena-verified dataset")


def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """Load webarena-verified config for URL resolution.
    
    Args:
        config_path: Path to config file. Uses default if not provided.
        
    Returns:
        Config dictionary with environment URLs
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if path.exists():
        with path.open() as f:
            return json.load(f)
    logger.warning(f"Config file not found at {path}, using empty config")
    return {}


def resolve_url(raw_url: str, config: dict[str, Any]) -> str:
    """Resolve placeholder URLs like __SHOPPING__ to actual URLs.
    
    Args:
        raw_url: URL potentially containing placeholders like __SHOPPING__
        config: Config dictionary with environments mapping
        
    Returns:
        Resolved URL with actual server addresses
    """
    environments = config.get("environments", {})
    for placeholder, env_config in environments.items():
        if raw_url == placeholder:
            urls = env_config.get("urls", [])
            if urls:
                return urls[0]
        elif raw_url.startswith(f"{placeholder}/"):
            urls = env_config.get("urls", [])
            if urls:
                suffix = raw_url[len(placeholder):]
                return urls[0].rstrip("/") + suffix
    return raw_url


def get_task_type(task_data: dict[str, Any]) -> str:
    """Extract expected task_type from task evaluation config.
    
    Args:
        task_data: Task data from webarena-verified dataset
        
    Returns:
        Task type string: "RETRIEVE", "MUTATE", or "NAVIGATE"
    """
    eval_config = task_data.get("eval", [])
    if eval_config:
        expected = eval_config[0].get("expected", {})
        return expected.get("task_type", "RETRIEVE").upper()
    return "RETRIEVE"


def create_agent_response(
    task_type: str,
    status: str,
    retrieved_data: Any = None,
    error_details: Optional[str] = None,
) -> dict[str, Any]:
    """Create structured agent response for WebArena-Verified.
    
    Args:
        task_type: One of "RETRIEVE", "MUTATE", "NAVIGATE"
        status: One of "SUCCESS", "NOT_FOUND_ERROR", etc.
        retrieved_data: Data to return (should be list for RETRIEVE)
        error_details: Error message if status is not SUCCESS
        
    Returns:
        Structured agent response dictionary
    """
    return {
        "task_type": task_type,
        "status": status,
        "retrieved_data": retrieved_data,
        "error_details": error_details,
    }


def _create_dummy_har(path: Path) -> None:
    """Create minimal HAR file for agent-response-only tasks.
    
    Args:
        path: Path where to save the dummy HAR file
    """
    dummy = {
        "log": {
            "version": "1.2",
            "creator": {"name": "dummy", "version": "1.0"},
            "entries": [{
                "request": {
                    "method": "GET",
                    "url": "http://dummy.local",
                    "headers": [],
                    "queryString": [],
                    "cookies": [],
                    "bodySize": 0,
                },
                "response": {
                    "status": 200,
                    "statusText": "OK",
                    "headers": [],
                    "cookies": [],
                    "content": {"size": 0, "mimeType": "text/plain"},
                    "bodySize": 0,
                },
                "cache": {},
                "timings": {"send": 0, "wait": 0, "receive": 0},
            }]
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(dummy, f)


def trim_har_file(har_org: Path, har_trimmed: Path) -> dict[str, Any]:
    """Trim HAR file to remove unnecessary data.
    
    Uses webarena-verified's trim_network_logs utility.
    
    Args:
        har_org: Path to original HAR file
        har_trimmed: Path where to save trimmed HAR file
        
    Returns:
        Stats dictionary from trimming operation
    """
    # Add webarena-verified to path if needed
    webarena_src = _WEBARENA_VERIFIED_DIR / "src"
    if str(webarena_src) not in sys.path:
        sys.path.insert(0, str(webarena_src))
    
    from webarena_verified.core.utils.trim_network_logs import trim_har_file as _trim
    
    return _trim(har_org, har_trimmed)


def run_verified_evaluation(
    task_id: int,
    exp_dir: Path,
    config_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run WebArena-Verified evaluation.
    
    Args:
        task_id: Task ID to evaluate
        exp_dir: Experiment directory containing agent_response.json and network.har
        config_path: Path to webarena-verified config file
        
    Returns:
        Dictionary with 'score', 'status', and 'evaluators_results'
    """
    # Add webarena-verified to path if needed
    webarena_src = _WEBARENA_VERIFIED_DIR / "src"
    if str(webarena_src) not in sys.path:
        sys.path.insert(0, str(webarena_src))
    
    from webarena_verified.api import WebArenaVerified
    
    exp_dir = Path(exp_dir)
    
    # Paths
    agent_response = exp_dir / "agent_response.json"
    har_org = exp_dir / "network.har"
    har_trimmed = exp_dir / "network_trimmed.har"
    eval_result_path = exp_dir / "eval_result.json"
    
    # Check required files
    if not agent_response.exists():
        logger.error(f"agent_response.json not found at {agent_response}")
        return {"score": 0.0, "status": "error", "error": "agent_response.json not found"}
    
    # Trim HAR if exists
    if har_org.exists():
        try:
            trim_har_file(har_org, har_trimmed)
            logger.info(f"HAR trimmed: {har_org} -> {har_trimmed}")
        except Exception as e:
            logger.warning(f"Could not trim HAR: {e}, using original")
            har_trimmed = har_org
    else:
        # Create dummy HAR for agent_response-only evaluation
        logger.warning(f"HAR file not found at {har_org}, creating dummy HAR")
        har_trimmed = exp_dir / "network_dummy.har"
        _create_dummy_har(har_trimmed)
    
    # Initialize evaluator
    config = config_path or _DEFAULT_CONFIG_PATH
    if not config.exists():
        logger.warning(f"Config file not found at {config}")
    
    try:
        wa = WebArenaVerified(config=config)
        
        # Run evaluation
        result = wa.evaluate_task(
            task_id=task_id,
            agent_response=agent_response,
            network_trace=har_trimmed,
        )
        
        # Save evaluation result
        eval_result_dict = result.model_dump(mode='json')
        with eval_result_path.open("w") as f:
            json.dump(eval_result_dict, f, indent=2)
        logger.info(f"Evaluation result saved to {eval_result_path}")
        
        # Build return value
        return_value = {
            "score": result.score,
            "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
            "task_id": task_id,
        }
        
        if result.evaluators_results:
            return_value["evaluators_results"] = [
                {
                    "evaluator_name": er.evaluator_name,
                    "score": er.score,
                    "status": er.status.value if hasattr(er.status, 'value') else str(er.status),
                }
                for er in result.evaluators_results
            ]
        
        return return_value
        
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {"score": 0.0, "status": "error", "error": str(e)}


# Site credentials for WebArena environments
SITE_CREDENTIALS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
    "map": {},  # No auth needed
}


def get_login_info(sites: list[str]) -> str:
    """Build login information string for task prompt.
    
    Args:
        sites: List of site names from task data
        
    Returns:
        Formatted login info string for inclusion in prompt
    """
    site_creds = []
    for site in sites:
        creds = SITE_CREDENTIALS.get(site)
        if creds and creds.get("username"):
            site_creds.append(
                f"- Site `{site}`: username: `{creds['username']}`, password: `{creds['password']}`"
            )
    
    if site_creds:
        login_details = "\n".join(site_creds)
        return (
            "\n\n**Authentication:** You may already be logged in. If re-authentication is needed:\n"
            f"{login_details}"
        )
    return ""

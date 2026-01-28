# HAR Recording Integration Plan for BrowserGym

## Overview

This plan integrates HAR recording capability into the BrowserGym framework used by `agent-skill-induction`, enabling WebArena-Verified network event-based evaluation.

## Files to Modify

### 1. `browsergym/experiments/src/browsergym/experiments/loop.py`

**Changes:**
- Add `record_har` parameter to `EnvArgs`
- Pass HAR recording config to `pw_context_kwargs`
- Add post-run HAR file handling

```python
# In EnvArgs dataclass (around line 34)
@dataclass
class EnvArgs:
    task_name: str
    task_seed: Optional[int] = None
    max_steps: Optional[int] = None
    headless: bool = True
    record_video: bool = False
    record_har: bool = False  # NEW
    wait_for_user_message: bool = False
    viewport: Optional[dict] = None
    slow_mo: Optional[int] = None
    storage_state: Optional[str | Path | dict] = None
    task_kwargs: Optional[dict] = None
```

```python
# In EnvArgs.make_env method
def make_env(self, action_mapping, exp_dir):
    pw_context_kwargs = {}
    
    # HAR recording
    if self.record_har:
        har_path = exp_dir / "network.har"
        pw_context_kwargs["record_har_path"] = str(har_path)
        pw_context_kwargs["record_har_omit_content"] = True
    
    return gym.make(
        self.task_name,
        ...,
        pw_context_kwargs=pw_context_kwargs,
    )
```

### 2. `browsergym/core/src/browsergym/core/env.py`

**No changes needed!** The `pw_context_kwargs` already passes through to `browser.new_context()`.

### 3. `asi/run_demo.py`

**Changes:**
- Add `--record-har` argument
- Add `--eval-verified` argument
- Add post-run evaluation integration

```python
# Add new arguments
parser.add_argument("--record-har", action="store_true", 
                    help="Record HAR file for network trace evaluation")
parser.add_argument("--eval-verified", action="store_true",
                    help="Run WebArena-Verified evaluation after completion")

# Update EnvArgs
env_args = EnvArgs(
    task_name=args.task_name,
    ...
    record_har=args.record_har or args.eval_verified,
)

# Post-run: HAR trimming and evaluation
if args.eval_verified:
    from webarena_verified.core.utils.trim_network_logs import trim_har_file
    har_org = exp_dir / "network.har"
    har_trimmed = exp_dir / "network_trimmed.har"
    if har_org.exists():
        trim_har_file(har_org, har_trimmed)
    
    # Run evaluation
    from webarena_verified.api import WebArenaVerified
    wa = WebArenaVerified(config=config_path)
    result = wa.evaluate_task(
        task_id=task_id,
        agent_response=exp_dir / "agent_response.json",
        network_trace=har_trimmed,
    )
```

### 4. Create `asi/webarena_verified_utils.py` (NEW FILE)

Utility functions for WebArena-Verified integration:

```python
"""WebArena-Verified utilities for BrowserGym integration."""
import json
from pathlib import Path
from typing import Any, Optional

def load_verified_task(task_name: str) -> dict[str, Any]:
    """Load task from webarena-verified dataset.
    
    Args:
        task_name: Format "webarena_verified.{task_id}" or just task_id
    """
    # Parse task_id from task_name
    if task_name.startswith("webarena_verified."):
        task_id = int(task_name.split(".")[-1])
    elif task_name.startswith("webarena."):
        task_id = int(task_name.split(".")[-1])
    else:
        task_id = int(task_name)
    
    dataset_path = Path("../webarena-verified/assets/dataset/webarena-verified.json")
    with dataset_path.open() as f:
        tasks = json.load(f)
    
    for task in tasks:
        if task["task_id"] == task_id:
            return task
    
    raise ValueError(f"Task {task_id} not found")


def resolve_url(raw_url: str, config: dict[str, Any]) -> str:
    """Resolve placeholder URLs like __SHOPPING__ to actual URLs."""
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


def create_agent_response(task_type: str, status: str, 
                          retrieved_data: Any = None, 
                          error_details: str = None) -> dict[str, Any]:
    """Create structured agent response for WebArena-Verified."""
    return {
        "task_type": task_type,
        "status": status,
        "retrieved_data": retrieved_data,
        "error_details": error_details,
    }


def run_verified_evaluation(
    task_id: int,
    exp_dir: Path,
    config_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run WebArena-Verified evaluation.
    
    Returns:
        dict with 'score', 'status', and 'evaluators_results'
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "webarena-verified/src"))
    
    from webarena_verified.api import WebArenaVerified
    from webarena_verified.core.utils.trim_network_logs import trim_har_file
    
    # Paths
    agent_response = exp_dir / "agent_response.json"
    har_org = exp_dir / "network.har"
    har_trimmed = exp_dir / "network_trimmed.har"
    
    # Trim HAR if exists
    if har_org.exists():
        try:
            trim_har_file(har_org, har_trimmed)
        except Exception as e:
            print(f"Warning: Could not trim HAR: {e}")
            har_trimmed = har_org
    else:
        # Create dummy HAR for agent_response-only evaluation
        har_trimmed = exp_dir / "network_dummy.har"
        _create_dummy_har(har_trimmed)
    
    # Initialize evaluator
    config = config_path or Path(__file__).parent.parent.parent / "verified_config.json"
    wa = WebArenaVerified(config=config)
    
    # Run evaluation
    result = wa.evaluate_task(
        task_id=task_id,
        agent_response=agent_response,
        network_trace=har_trimmed,
    )
    
    return {
        "score": result.score,
        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
        "evaluators_results": [
            {
                "evaluator_name": er.evaluator_name,
                "score": er.score,
                "status": er.status.value if hasattr(er.status, 'value') else str(er.status),
            }
            for er in result.evaluators_results
        ] if result.evaluators_results else [],
    }


def _create_dummy_har(path: Path) -> None:
    """Create minimal HAR file for agent-response-only tasks."""
    dummy = {
        "log": {
            "version": "1.2",
            "creator": {"name": "dummy", "version": "1.0"},
            "entries": [{
                "request": {
                    "method": "GET", "url": "http://dummy.local",
                    "headers": [], "queryString": [], "cookies": [], "bodySize": 0
                },
                "response": {
                    "status": 200, "statusText": "OK",
                    "headers": [], "cookies": [],
                    "content": {"size": 0, "mimeType": "text/plain"},
                    "bodySize": 0
                },
                "cache": {},
                "timings": {"send": 0, "wait": 0, "receive": 0}
            }]
        }
    }
    with path.open("w") as f:
        json.dump(dummy, f)
```

## Implementation Steps

### Step 1: Modify `loop.py`
- Add `record_har` to EnvArgs
- Implement HAR config passing in `make_env`

### Step 2: Create `webarena_verified_utils.py`
- Add utility functions for loading tasks
- Add evaluation wrapper

### Step 3: Modify `run_demo.py`
- Add CLI arguments
- Integrate evaluation post-run

### Step 4: Test with single task

```bash
# Test command
cd /home/demiw/Code-Web-Agent/agent-skill-induction
MODEL=azure/gpt-4o-mini
python asi/run_demo.py \
    --model_name $MODEL \
    --task_name "webarena.117" \
    --websites "shopping" \
    --headless \
    --record-har \
    --eval-verified
```

## Testing Checklist

- [ ] HAR file is created in experiment directory
- [ ] HAR file contains network events
- [ ] HAR trimming works correctly
- [ ] Evaluation runs successfully
- [ ] Score is returned correctly

## Notes

- The integration uses Playwright's native HAR recording (via `record_har_path`), not CDP-based recording
- This is simpler than the OpenHands approach but may capture slightly different data
- For NetworkEventEvaluator, the key headers (`sec-fetch-dest`, `sec-fetch-mode`) should be present

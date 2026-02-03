"""
WebArena-Verified Task for BrowserGym

This module provides a task class that integrates WebArena-Verified evaluation
into browsergym, similar to how GenericWebArenaTask works with the original webarena.

The key difference is that validation uses WebArena-Verified's AgentResponseEvaluator
instead of the original webarena evaluator.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import playwright.sync_api
from browsergym.core.task import AbstractBrowserTask

# Import the original WebArena instance for URL resolution and login
from browsergym.webarena.instance import WebArenaInstance

logger = logging.getLogger(__name__)

# Paths
WEBARENA_VERIFIED_DATASET = Path(
    "/home/demiw/webarena-verified/assets/dataset/webarena-verified.json"
)
WEBARENA_VERIFIED_CONFIG = Path("/home/demiw/webarena_verified_quickstart/config.json")


class VerifiedWebArenaTask(AbstractBrowserTask):
    """
    BrowserGym task for WebArena-Verified evaluation.

    Similar to GenericWebArenaTask but uses WebArena-Verified's evaluator
    and expects structured JSON responses via save_agent_response().
    """

    def __init__(
        self,
        seed: int,
        task_id: Optional[int] = None,
        exp_dir: Optional[str | Path] = None,
    ) -> None:
        super().__init__(seed)

        # Task properties (same as GenericWebArenaTask)
        self.viewport = {"width": 1280, "height": 720}
        self.slow_mo = 1000  # ms
        self.timeout = 10000  # ms

        # Use the same WebArena instance for URL resolution
        self.webarena_instance = WebArenaInstance()

        if task_id is None:
            raise ValueError("task_id must be provided")

        self.task_id = task_id
        self.config = self._load_task_config(task_id)
        self.agent_response_file = None
        self.exp_dir = Path(exp_dir) if exp_dir else None

    def _load_task_config(self, task_id: int) -> dict:
        """Load task from webarena-verified dataset."""
        with open(WEBARENA_VERIFIED_DATASET, "r") as f:
            all_tasks = json.load(f)

        for task in all_tasks:
            if task["task_id"] == task_id:
                return task

        raise ValueError(f"Task {task_id} not found in webarena-verified dataset")

    def _get_expected_task_type(self) -> str:
        """Get expected task_type from evaluation config."""
        eval_config = self.config.get("eval", [])
        if eval_config:
            expected = eval_config[0].get("expected", {})
            return expected.get("task_type", "RETRIEVE").upper()
        return "RETRIEVE"

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        """Set up the task - same pattern as GenericWebArenaTask."""

        # Clean up stale files from previous runs (critical for batch execution)
        try:
            Path("agent_response.json").unlink(missing_ok=True)
        except Exception:
            pass

        # Create temp file for agent response
        self.agent_response_file = (
            Path(tempfile.gettempdir()) / f"agent_response_{self.task_id}.json"
        )

        # Clean up stale temp file if exists
        try:
            if self.agent_response_file.exists():
                self.agent_response_file.unlink()
        except Exception:
            pass

        # Authenticate for each site (same as GenericWebArenaTask)
        for site in self.config.get("sites", []):
            self.webarena_instance.ui_login(site=site, page=page)

        # Set geolocation
        geolocation = self.config.get("geolocation")
        if geolocation:
            page.context.set_geolocation(geolocation)

        # Navigate to start URL(s)
        start_urls = self.config.get("start_urls", [])
        if not start_urls:
            start_url = self.config.get("start_url", "")
            if start_url:
                start_urls = start_url.split(" |AND| ")

        for i, url in enumerate(start_urls):
            # Resolve URL placeholders
            for pattern, url_key in {
                "__GITLAB__": "gitlab",
                "__REDDIT__": "reddit",
                "__SHOPPING__": "shopping",
                "__SHOPPING_ADMIN__": "shopping_admin",
                "__WIKIPEDIA__": "wikipedia",
                "__MAP__": "map",
            }.items():
                if url.startswith(pattern):
                    base = self.webarena_instance.urls.get(url_key, "")
                    if base:
                        url = url.replace(pattern, base.rstrip("/"))
                    break

            page.goto(url)
            if i < len(start_urls) - 1:
                page = page.context.new_page()

        # Build goal with response format instructions
        goal = self._build_goal()

        return goal, {}

    def _build_goal(self) -> str:
        """Build goal with WebArena-Verified response format."""
        intent = self.config["intent"]
        task_type = self._get_expected_task_type()

        # Add response format instructions to the goal
        goal = f"""{intent}

When you complete the task, call save_agent_response() with your response:
- task_type: "{task_type}"
- status: "SUCCESS" or error code (NOT_FOUND_ERROR, PERMISSION_DENIED_ERROR, etc.)
- retrieved_data: list of results for RETRIEVE, null otherwise
- error_details: null for SUCCESS, explanation otherwise

Example: save_agent_response({{"task_type": "{task_type}", "status": "SUCCESS", "retrieved_data": {"[...]" if task_type == "RETRIEVE" else "null"}, "error_details": null}})
"""
        return goal

    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        raise NotImplementedError

    @classmethod
    def get_task_id(cls):
        raise NotImplementedError

    def teardown(self) -> None:
        """Clean up resources."""
        # Clean up agent_response.json
        try:
            Path("agent_response.json").unlink(missing_ok=True)
        except Exception:
            pass

        # Clean up temp file
        try:
            if self.agent_response_file and self.agent_response_file.exists():
                self.agent_response_file.unlink()
        except Exception:
            pass

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:
        """
        Validate using WebArena-Verified evaluator.

        Similar to GenericWebArenaTask.validate() but uses WebArena-Verified API.
        """
        import urllib.parse

        # Check authorized URLs (same as GenericWebArenaTask)
        authorized_locations = ["newtab", ""] + [
            urllib.parse.urlparse(url).netloc
            for url in [
                *self.webarena_instance.urls.values(),
                self.webarena_instance.home_url,
            ]
        ]
        for open_page in page.context.pages:
            page_location = urllib.parse.urlparse(open_page.url).netloc
            if page_location and page_location not in authorized_locations:
                return 0, True, "", {"error": "Unauthorized url, terminating task"}

        # Check if agent called save_agent_response
        # Look for the response in chat messages or in the temp file
        agent_response = None

        # Check chat messages for save_agent_response call
        if chat_messages:
            for msg in reversed(chat_messages):
                if msg.get("role") == "assistant":
                    text = msg.get("message", "")
                    if "save_agent_response" in text:
                        agent_response = self._extract_response_from_action(text)
                        if agent_response:
                            break

        # Also check if file exists (written by save_agent_response action)
        if (
            not agent_response
            and self.agent_response_file
            and self.agent_response_file.exists()
        ):
            try:
                with open(self.agent_response_file, "r") as f:
                    agent_response = json.load(f)
            except:
                pass

        # Also check current working directory
        cwd_response = Path("agent_response.json")
        if not agent_response and cwd_response.exists():
            try:
                with open(cwd_response, "r") as f:
                    agent_response = json.load(f)
            except:
                pass

        if not agent_response:
            # Agent hasn't provided response yet
            return 0, False, "", {}

        # Resolve HAR path from current experiment directory
        har_path = None
        if self.exp_dir:
            candidate = self.exp_dir / "network.har"
            if candidate.exists():
                har_path = candidate
        if not har_path:
            exp_dir = os.environ.get("BROWSERGYM_EXP_DIR")
            if exp_dir:
                candidate = Path(exp_dir) / "network.har"
                if candidate.exists():
                    har_path = candidate
        if not har_path:
            results_root = os.environ.get("BROWSERGYM_RESULTS_ROOT")
            if results_root:
                task_root = Path(results_root) / str(self.task_id)
                if task_root.exists():
                    run_dirs = [p for p in task_root.iterdir() if p.is_dir()]
                    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    for run_dir in run_dirs:
                        candidate = run_dir / "network.har"
                        if candidate.exists():
                            har_path = candidate
                            break

        # Run WebArena-Verified evaluation
        try:
            score = self._run_verified_evaluation(agent_response, har_path=har_path)
            return score, True, "", {"agent_response": agent_response, "score": score}
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return 0, True, "", {"error": str(e)}

    def _extract_response_from_action(self, action_text: str) -> dict | None:
        """Extract JSON response from save_agent_response() call."""
        import re

        match = re.search(
            r"save_agent_response\s*\(\s*(\{[\s\S]*?\})\s*\)", action_text
        )
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                try:
                    return eval(match.group(1))
                except:
                    pass
        return None

    def _run_verified_evaluation(
        self, agent_response: dict, har_path: Path | None = None
    ) -> float:
        """Run WebArena-Verified evaluation and return score.

        Args:
            agent_response: Agent's structured response dictionary
            har_path: Optional path to HAR file. If not provided, uses dummy HAR.
        """
        try:
            from webarena_verified.api import WebArenaVerified
            import tempfile

            config = (
                WEBARENA_VERIFIED_CONFIG if WEBARENA_VERIFIED_CONFIG.exists() else None
            )
            wa = WebArenaVerified(config=config)

            # Check if we have a real HAR file
            network_trace = None
            temp_har_path = None

            if har_path and har_path.exists():
                # Use provided HAR file
                network_trace = har_path
                logger.info(
                    f"Using HAR file for evaluation: {har_path} ({har_path.stat().st_size:,} bytes)"
                )
            else:
                # Create valid dummy HAR file with one entry
                # webarena-verified requires at least one entry
                logger.warning("No HAR file provided, using dummy HAR for evaluation")
                dummy_har = {
                    "log": {
                        "version": "1.2",
                        "creator": {"name": "Playwright", "version": "1.0.0"},
                        "browser": {"name": "chromium", "version": "100.0.0"},
                        "entries": [
                            {
                                "startedDateTime": "2025-01-01T00:00:00.000Z",
                                "time": 0.1,
                                "request": {
                                    "method": "GET",
                                    "url": "http://localhost/dummy",
                                    "httpVersion": "HTTP/1.1",
                                    "cookies": [],
                                    "headers": [],
                                    "queryString": [],
                                    "headersSize": -1,
                                    "bodySize": -1,
                                },
                                "response": {
                                    "status": 200,
                                    "statusText": "OK",
                                    "httpVersion": "HTTP/1.1",
                                    "cookies": [],
                                    "headers": [],
                                    "content": {"size": -1, "mimeType": "text/html"},
                                    "headersSize": -1,
                                    "bodySize": -1,
                                    "redirectURL": "",
                                },
                                "cache": {},
                                "timings": {"send": -1, "wait": -1, "receive": 0.1},
                            }
                        ],
                    }
                }

                # Write dummy HAR to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".har", delete=False
                ) as f:
                    json.dump(dummy_har, f)
                    temp_har_path = Path(f.name)
                    network_trace = temp_har_path

            try:
                result = wa.evaluate_task(
                    task_id=self.task_id,
                    agent_response=agent_response,
                    network_trace=network_trace,
                )

                logger.info(
                    f"Evaluation result: score={result.score}, status={result.status}"
                )
                if result.evaluators_results:
                    for er in result.evaluators_results:
                        logger.info(
                            f"  - {er.evaluator_name}: {er.status} (score: {er.score})"
                        )

                return result.score
            finally:
                # Clean up temp file only
                if temp_har_path and temp_har_path.exists():
                    temp_har_path.unlink()

        except ImportError as e:
            logger.warning(
                f"webarena_verified not available ({e}), using basic validation"
            )
            if agent_response.get("status") == "SUCCESS":
                return 1.0
            return 0.0
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback

            traceback.print_exc()
            return 0.0


def register_verified_tasks():
    """Register WebArena-Verified tasks with gymnasium."""
    import gymnasium as gym

    with open(WEBARENA_VERIFIED_DATASET, "r") as f:
        all_tasks = json.load(f)

    for task in all_tasks:
        task_id = task["task_id"]
        env_id = f"browsergym/webarena_verified.{task_id}"

        if env_id not in gym.envs.registry:
            gym.register(
                id=env_id,
                entry_point="browsergym.core.env:BrowserEnv",
                kwargs={
                    "task_entrypoint": VerifiedWebArenaTask,
                    "task_kwargs": {"task_id": task_id},
                },
            )

    logger.info(f"Registered {len(all_tasks)} webarena_verified tasks")

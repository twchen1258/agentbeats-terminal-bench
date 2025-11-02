"""
Green Agent for evaluating other agents on terminal-bench.
This agent receives evaluation requests via A2A protocol and runs terminal-bench harness.
"""

import json
import logging
import re
import tomllib
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import Any

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import AgentCard, Part, TextPart, TaskState
from a2a.utils import new_task, new_agent_text_message
from a2a.server.tasks import TaskUpdater

from terminal_bench.harness.harness import Harness
from terminal_bench.harness.models import BenchmarkResults

from src.config import settings
from src.config.settings import ConfigurationError

logger = logging.getLogger(__name__)


class TerminalBenchGreenAgentExecutor(AgentExecutor):
    """
    Executes terminal-bench evaluation when receiving requests via A2A.
    """

    def __init__(self):
        self.evaluation_history = []
        logger.info("TerminalBenchGreenAgentExecutor initialized")

    def parse_task_config(self, user_input: str) -> dict[str, Any]:
        """
        Parse task configuration from user input.
        Extracts JSON config from <task_config> tags.
        """
        match = re.search(r"<task_config>(.*?)</task_config>", user_input, re.DOTALL)
        if match:
            config_json = match.group(1).strip()
            return json.loads(config_json)
        try:
            return json.loads(user_input)
        except json.JSONDecodeError:
            raise ValueError("Could not parse task configuration from user input")

    def run_terminal_bench_evaluation(self, config: dict[str, Any]) -> BenchmarkResults:
        """
        Run terminal-bench harness with the given configuration.

        Args:
            config: Dictionary containing evaluation configuration
                - task_ids: List of task IDs to run
                - dataset_name: Name of the dataset to run
                - dataset_version: Version of the dataset to run
                - white_agent_url: URL of the agent being evaluated
                - n_attempts: Number of attempts per task
                - n_concurrent_trials: Number of concurrent trials
                - timeout_multiplier: Timeout multiplier
        """
        logger.info(f"Starting terminal-bench evaluation with config: {config}")

        # Create output directory for this evaluation run
        run_id = f"green_agent_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = Path(settings.eval_output_path)
        output_path.mkdir(exist_ok=True)

        # Extract configuration (all required, no fallbacks)
        white_agent_url = config.get("white_agent_url")
        dataset_name = config.get("dataset_name")
        dataset_version = config.get("dataset_version")
        task_ids = config.get("task_ids")
        n_attempts = config.get("n_attempts")
        n_concurrent_trials = config.get("n_concurrent_trials")
        timeout_multiplier = config.get("timeout_multiplier")

        # Log configuration
        logger.info(f"Evaluating agent at: {white_agent_url}")
        logger.info(f"Dataset: {dataset_name} (version: {dataset_version})")
        logger.info(f"Task IDs: {task_ids}")

        # Create harness instance
        harness_kwargs = {
            "output_path": output_path,
            "run_id": run_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "agent_import_path": "src.adapters.a2a_adapter:A2AAdapter",
            "agent_kwargs": {"agent_url": white_agent_url},
            "task_ids": [str(tid) for tid in task_ids] if task_ids else None,
            "n_attempts": n_attempts,
            "n_concurrent_trials": n_concurrent_trials,
            "global_timeout_multiplier": timeout_multiplier,
            "cleanup": settings.eval_cleanup,
            "log_level": getattr(logging, settings.log_level),
        }
        harness = Harness(**harness_kwargs)

        # Run the evaluation
        logger.info("Running terminal-bench harness...")
        results = harness.run()
        logger.info(f"Evaluation complete. Accuracy: {results.accuracy:.2%}")
        logger.info(f"Results saved to: {output_path / run_id}")

        return results

    def format_results_message(
        self, results: BenchmarkResults, config: dict[str, Any]
    ) -> str:
        """Format evaluation results into a human-readable message."""
        # Load scoring configuration from settings
        TASK_DIFFICULTY_MAP = settings.task_difficulty_map
        DIFFICULTY_WEIGHTS = settings.difficulty_weights

        category_scores = {
            "easy": [],
            "medium": [],
            "hard": [],
            "unknown": [],
        }
        task_scores_list = []
        failure_mode_counts = {}

        base_output_dir = Path(settings.eval_output_path)

        for result in results.results:
            parser_results = {}
            task_id = result.task_id

            if not result.recording_path:
                logger.warning(
                    f"No recording_path for task {task_id}, cannot load parser_results."
                )
            else:
                try:
                    trial_dir = (
                        base_output_dir / Path(result.recording_path).parent.parent
                    )
                    results_json_path = trial_dir / "results.json"

                    if results_json_path.exists():
                        with open(results_json_path, "r") as f:
                            trial_data = json.load(f)

                        if "parser_results" in trial_data and isinstance(
                            trial_data["parser_results"], dict
                        ):
                            parser_results = trial_data["parser_results"]
                        else:
                            logger.warning(
                                f"No 'parser_results' dict found in {results_json_path}"
                            )
                    else:
                        logger.warning(f"results.json not found at {results_json_path}")
                except Exception as e:
                    logger.error(
                        f"Error loading {results_json_path} for task {task_id}: {e}",
                        exc_info=True,
                    )

            num_tests = 0
            num_passed = 0
            test_case_score_component = 0.0

            if parser_results:
                num_tests = len(parser_results)
                if num_tests > 0:
                    num_passed = sum(
                        1 for status in parser_results.values() if status == "passed"
                    )
                    test_case_score_component = 0.5 * (num_passed / num_tests)

            resolved_score_component = 0.5 if result.is_resolved else 0.0
            task_score = test_case_score_component + resolved_score_component

            difficulty = TASK_DIFFICULTY_MAP.get(task_id, "unknown")
            category_scores[difficulty].append(task_score)

            if not result.is_resolved:
                failure_mode = result.failure_mode
                failure_mode_key = "unknown"
                if failure_mode:
                    failure_mode_key = (
                        failure_mode.value
                        if hasattr(failure_mode, "value")
                        else str(failure_mode)
                    )
                
                if failure_mode_key == "unset":
                    failure_mode_key = "other (unset)"

                failure_mode_counts[failure_mode_key] = failure_mode_counts.get(
                    failure_mode_key, 0
                ) + 1

            task_scores_list.append(
                {
                    "id": task_id,
                    "score": task_score,
                    "is_resolved": result.is_resolved,
                    "tests_passed": num_passed,
                    "tests_total": num_tests,
                    "failure_mode": result.failure_mode,
                    "total_input_tokens": result.total_input_tokens,
                    "total_output_tokens": result.total_output_tokens,
                }
            )

        avg = lambda scores: (
            (sum(scores) / len(scores), len(scores)) if scores else (0.0, 0)
        )

        easy_avg, easy_count = avg(category_scores["easy"])
        medium_avg, medium_count = avg(category_scores["medium"])
        hard_avg, hard_count = avg(category_scores["hard"])
        unknown_avg, unknown_count = avg(category_scores["unknown"])

        total_weighted_score = (
            (sum(category_scores["easy"]) * DIFFICULTY_WEIGHTS["easy"])
            + (sum(category_scores["medium"]) * DIFFICULTY_WEIGHTS["medium"])
            + (sum(category_scores["hard"]) * DIFFICULTY_WEIGHTS["hard"])
            + (sum(category_scores["unknown"]) * DIFFICULTY_WEIGHTS["unknown"])
        )

        total_possible_weight = (
            (easy_count * DIFFICULTY_WEIGHTS["easy"])
            + (medium_count * DIFFICULTY_WEIGHTS["medium"])
            + (hard_count * DIFFICULTY_WEIGHTS["hard"])
            + (unknown_count * DIFFICULTY_WEIGHTS["unknown"])
        )

        if total_possible_weight == 0:
            weighted_overall_avg = 0.0
        else:
            weighted_overall_avg = total_weighted_score / total_possible_weight

        overall_count = easy_count + medium_count + hard_count + unknown_count

        failure_summary_message = ""
        if failure_mode_counts:
            failure_summary_message = "\nFailure Mode Summary:\n"
            sorted_failures = sorted(
                failure_mode_counts.items(), key=lambda item: item[1], reverse=True
            )
            for mode, count in sorted_failures:
                failure_summary_message += f"- {mode}: {count}\n"

        message = f"""
Terminal-Bench Evaluation Results
=====================================
(Weighting: Easy=1, Medium=2, Hard=3)

Evaluation Summary:
- Overall Score: {weighted_overall_avg:.2%}
- Resolved: {results.n_resolved}/{overall_count}
- Unresolved: {results.n_unresolved}/{overall_count}

Scores by Difficulty (Unweighted Avg):
- Easy:   {easy_avg:.2%}
- Medium: {medium_avg:.2%}
- Hard:   {hard_avg:.2%}
"""
        if unknown_count > 0:
            message += f"- Unknown: {unknown_avg:.2%} ({unknown_count} tasks) -- *Task ID not in TASK_DIFFICULTY_MAP*\n"

        message += failure_summary_message

        if results.pass_at_k:
            message += "\nPass@k Metrics (based on is_resolved):\n"
            for k, score in results.pass_at_k.items():
                if score is not None:
                    message += f"- Pass@{k}: {score:.2%}\n"
            message += "\n"

        # Add per-task results
        message += "Task Results:\n"
        message += "-" * 60 + "\n"

        for task in task_scores_list:
            status = "✓" if task["is_resolved"] else "✗"
            message += f"{status} Score: {task['score']:.2%} - {task['id']} (Tests: {task['tests_passed']}/{task['tests_total']})\n"

            if not task["is_resolved"] and task["failure_mode"]:
                failure_mode_val = (
                    task["failure_mode"].value
                    if hasattr(task["failure_mode"], "value")
                    else task["failure_mode"]
                )
                message += f"      Failure Mode: {failure_mode_val}\n"
            if task["total_input_tokens"] or task["total_output_tokens"]:
                message += f"      Tokens: {task['total_input_tokens'] or 0} in, {task['total_output_tokens'] or 0} out\n"

        message += "\n" + "=" * 60 + "\n"

        return message

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Execute the green agent - run terminal-bench evaluation.
        """
        logger.info("Green agent execute() called")

        # Create or get current task
        task = context.current_task
        if task is None:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        logger.info("Task created, sending initial status")

        # Set status to working
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Received evaluation request. Parsing configuration...\n",
                task.context_id,
                task.id,
            ),
        )

        logger.info("Initial status sent")

        try:
            # Parse task configuration from user input
            user_input = context.get_user_input()
            logger.info(f"Received user input: {user_input}")

            task_config = self.parse_task_config(user_input)
            logger.info(f"Parsed task config: {task_config}")

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Configuration parsed. Starting evaluation of agent at {task_config.get('white_agent_url')}...\n",
                    task.context_id,
                    task.id,
                ),
            )

            # Run terminal-bench evaluation
            results = self.run_terminal_bench_evaluation(task_config)

            # Store in history
            self.evaluation_history.append(
                {
                    "config": task_config,
                    "results": results,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            # Format results message
            results_message = self.format_results_message(results, task_config)

            # Send final response
            await updater.add_artifact(
                [Part(root=TextPart(text=results_message))],
                name="evaluation_results",
            )
            await updater.complete()

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Full traceback:\n{error_details}")
            error_message = (
                f"Error during evaluation: {str(e)}\n\nTraceback:\n{error_details}"
            )

            try:
                await updater.add_artifact(
                    [Part(root=TextPart(text=error_message))],
                    name="error",
                )
                await updater.update_status(
                    TaskState.failed,
                    new_agent_text_message(error_message, task.context_id, task.id),
                )
                await updater.complete()
            except Exception as update_error:
                logger.error(f"Failed to send error update: {update_error}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current evaluation (not implemented)."""
        raise NotImplementedError("cancel not supported")


def create_green_agent_app(agent_card_path: str) -> A2AStarletteApplication:
    """Create A2A application for the green agent."""

    # Load agent card
    with open(agent_card_path, "rb") as f:
        agent_card_data = tomllib.load(f)

    # Create A2A application
    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_data),
        http_handler=DefaultRequestHandler(
            agent_executor=TerminalBenchGreenAgentExecutor(),
            task_store=InMemoryTaskStore(),
        ),
    ).build()

    return app


def main():
    """Main entry point for the green agent."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format=settings.log_format,
    )

    # Configuration is validated automatically when properties are accessed
    logger.info("Starting green agent with config from config.toml")

    logger.info(
        f"Starting Terminal-Bench Green Agent on {settings.green_agent_host}:{settings.green_agent_port}"
    )
    logger.info(f"Using agent card: {settings.green_agent_card_path}")

    # Create and run app
    app = create_green_agent_app(settings.green_agent_card_path)
    uvicorn.run(app, host=settings.green_agent_host, port=settings.green_agent_port)


if __name__ == "__main__":
    main()

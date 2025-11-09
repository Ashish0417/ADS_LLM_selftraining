# # # Evaluation Module for ADS Framework
# # import logging
# # from typing import List, Dict, Any
# # import numpy as np
# # from utils import HeuristicScorer

# # logger = logging.getLogger(__name__)


# # class Evaluator:
# #     """Evaluation metrics for ADS framework"""
    
# #     def __init__(self, config):
# #         self.config = config
# #         self.use_heuristic = config.EVAL_CONFIG.get('use_heuristic_scoring', True)
    
# #     def evaluate_tasks(self, policy_model, tasks, api_executor=None):
# #         """Evaluate tasks using the policy model"""
# #         logger.info("Starting task evaluation...")
        
# #         results = {
# #             'tasks_completed': 0,
# #             'total_tasks': len(tasks),
# #             'average_score': 0,
# #             'win_rate': 0,
# #         }
        
# #         completed = 0
# #         total_score = 0
        
# #         for task in tasks:
# #             try:
# #                 instruction = task.get('instruction', '')
# #                 if not instruction:
# #                     continue
                
# #                 # Generate response using the policy model
# #                 response = policy_model.generate(instruction)
                
# #                 # Evaluate the response (THIS IS THE FIX)
# #                 reward = policy_model.evaluate_response(instruction, response)
                
# #                 total_score += reward
# #                 completed += 1
                
# #             except Exception as e:
# #                 logger.warning(f"Task evaluation failed: {e}")
import logging
from typing import List, Dict, Any, Optional

from utils import HeuristicScorer, answer_match_score

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator:
    """
    Evaluator compatible with `main.py`:
      - __init__(config=None, policy_model=None)
      - evaluate_tasks(policy_model=..., tasks=..., api_executor=...)

    Produces a serializable dict with per-task results and summary.
    """

    def __init__(self, config: Optional[Any] = None, policy_model: Optional[Any] = None):
        # store optional config and policy model
        self.config = config
        self.policy_model = policy_model
        # Heuristic scorer accessible via static method
        self.heuristic = HeuristicScorer

    def _score(self, instruction: str, response: str, reference: Optional[str]) -> float:
        """Return a score in [0,1]. Prefer exact/approx match when reference available."""
        if reference:
            try:
                return float(answer_match_score(reference, response))
            except Exception as e:
                logger.warning(f"[Evaluator] answer_match_score failed: {e}")

        # fallback to heuristic scorer
        try:
            return float(self.heuristic.score_response(instruction, response))
        except Exception:
            return 0.0

    def evaluate_tasks(self, policy_model: Optional[Any] = None, tasks: Optional[List[Dict[str, Any]]] = None,
                       api_executor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate a list of tasks.

        Args:
            policy_model: model with `.generate(instruction, context="")` and `.evaluate_response(...)` (optional)
            tasks: list of task dicts. Each may include 'instruction', 'context', 'response' (ground truth),
                   and/or 'before_response'/'after_response' when doing before/after scoring.
            api_executor: unused currently but accepted for API compatibility.

        Returns:
            dict with keys: total_tasks, tasks_completed, average_score, win_rate, task_results
        """
        if tasks is None:
            tasks = []

        # allow passing policy_model at call time or use stored one
        model = policy_model or self.policy_model

        results: Dict[str, Any] = {
            'total_tasks': len(tasks),
            'tasks_completed': 0,
            'average_score': 0.0,
            'win_rate': 0.0,
            'task_results': []
        }

        if not tasks:
            return results

        total_score = 0.0
        completed = 0

        for idx, t in enumerate(tasks, start=1):
            try:
                instruction = t.get('instruction', '') or ''
                context = t.get('context', '') or ''

                # If task already contains before/after responses, score them directly
                before_resp = t.get('before_response')
                after_resp = t.get('after_response')

                # If no explicit "before_response", generate with model (if available)
                if before_resp is None:
                    if model:
                        before_resp = model.generate(instruction, context=context) if hasattr(model, 'generate') else ''
                    else:
                        before_resp = ''

                # If no explicit "after_response", try to generate with context+retrieval if api_executor provided
                if after_resp is None:
                    if model:
                        after_resp = model.generate(f"{context}\n\n{instruction}" if context else instruction)
                    else:
                        after_resp = ''

                # prefer explicit ground truth 'answer' or 'response' fields for accurate scoring
                reference = t.get('answer') or t.get('response')

                before_score = self._score(instruction, before_resp or '', reference)
                after_score = self._score(instruction, after_resp or '', reference)

                score_gain = round(after_score - before_score, 6)

                task_result = {
                    'task_number': idx,
                    'instruction': instruction,
                    'context': context,
                    'before_response': before_resp,
                    'after_response': after_resp,
                    'before_score': round(before_score, 6),
                    'after_score': round(after_score, 6),
                    'score_gain': score_gain,
                    'response_improved': after_score > before_score
                }

                results['task_results'].append(task_result)

                total_score += after_score
                completed += 1

            except Exception as e:
                logger.warning(f"[Evaluator] Task {idx} evaluation failed: {e}")
                continue

        # finalize metrics
        results['tasks_completed'] = completed
        if completed > 0:
            avg = total_score / completed
            results['average_score'] = round(avg, 6)
            wins = sum(1 for tr in results['task_results'] if tr.get('after_score', 0) > 0.5)
            results['win_rate'] = round((wins / completed) * 100.0, 2)

        return results

    def summarize(self, evaluated: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Backwards compatible summary
        total = len(evaluated)
        if total == 0:
            return {
                "total_tasks": 0,
                "avg_before_score": 0.0,
                "avg_after_score": 0.0,
                "avg_improvement": 0.0,
                "tasks_improved": 0,
                "improvement_rate_percent": 0.0,
                "overall_improvement_percent": 0.0,
            }

        avg_before = sum(x.get("before_score", 0) for x in evaluated) / total
        avg_after = sum(x.get("after_score", 0) for x in evaluated) / total
        avg_improve = (avg_after - avg_before)
        improved = sum(1 for x in evaluated if x.get("response_improved"))
        improvement_rate = (improved / total) * 100.0
        overall_improvement_percent = (
            (avg_improve / abs(avg_before)) * 100.0 if avg_before != 0 else (100.0 if avg_after > 0 else 0.0)
        )

        return {
            "total_tasks": total,
            "avg_before_score": round(avg_before, 6),
            "avg_after_score": round(avg_after, 6),
            "avg_improvement": round(avg_improve, 6),
            "tasks_improved": improved,
            "improvement_rate_percent": round(improvement_rate, 2),
            "overall_improvement_percent": round(overall_improvement_percent, 2),
        }

        avg_before = sum(x.get("before_score", 0) for x in evaluated) / total
        avg_after = sum(x.get("after_score", 0) for x in evaluated) / total
        avg_improve = (avg_after - avg_before)
        improved = sum(1 for x in evaluated if x.get("response_improved"))
        improvement_rate = (improved / total) * 100.0
        overall_improvement_percent = (
            (avg_improve / abs(avg_before)) * 100.0 if avg_before != 0 else (100.0 if avg_after > 0 else 0.0)
        )

        return {
            "total_tasks": total,
            "avg_before_score": round(avg_before, 6),
            "avg_after_score": round(avg_after, 6),
            "avg_improvement": round(avg_improve, 6),
            "tasks_improved": improved,
            "improvement_rate_percent": round(improvement_rate, 2),
            "overall_improvement_percent": round(overall_improvement_percent, 2),
        }

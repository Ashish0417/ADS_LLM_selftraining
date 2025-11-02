# # Evaluation Module for ADS Framework
# import logging
# from typing import List, Dict, Any
# import numpy as np
# from utils import HeuristicScorer

# logger = logging.getLogger(__name__)


# class Evaluator:
#     """Evaluation metrics for ADS framework"""
    
#     def __init__(self, config):
#         self.config = config
#         self.use_heuristic = config.EVAL_CONFIG.get('use_heuristic_scoring', True)
    
#     def evaluate_tasks(self, policy_model, tasks, api_executor=None):
#         """Evaluate tasks using the policy model"""
#         logger.info("Starting task evaluation...")
        
#         results = {
#             'tasks_completed': 0,
#             'total_tasks': len(tasks),
#             'average_score': 0,
#             'win_rate': 0,
#         }
        
#         completed = 0
#         total_score = 0
        
#         for task in tasks:
#             try:
#                 instruction = task.get('instruction', '')
#                 if not instruction:
#                     continue
                
#                 # Generate response using the policy model
#                 response = policy_model.generate(instruction)
                
#                 # Evaluate the response (THIS IS THE FIX)
#                 reward = policy_model.evaluate_response(instruction, response)
                
#                 total_score += reward
#                 completed += 1
                
#             except Exception as e:
#                 logger.warning(f"Task evaluation failed: {e}")
#                 continue
        
#         # Calculate metrics
#         if completed > 0:
#             results['tasks_completed'] = completed
#             results['average_score'] = total_score / completed
#             results['win_rate'] = (results['average_score'] / 1.0) * 100
        
#         logger.info(f"Evaluation complete:")
#         logger.info(f"  - Tasks completed: {results['tasks_completed']}/{len(tasks)}")
#         logger.info(f"  - Average score: {results['average_score']:.4f}")
#         logger.info(f"  - Win rate: {results['win_rate']:.2f}%")
        
#         return results

    
#     def _reward_model_score(self, instruction: str, response: str) -> float:
#         """Score using reward model (not available on CPU, fallback to heuristic)"""
#         return HeuristicScorer.score_response(instruction, response)
    
#     @staticmethod
#     def compare_responses(original_response: str, improved_response: str) -> float:
#         """Compare two responses (returns improvement ratio 0-1)"""
#         orig_score = HeuristicScorer.score_response("", original_response)
#         improved_score = HeuristicScorer.score_response("", improved_response)
        
#         return (improved_score - orig_score) / (1.0 - orig_score) if orig_score < 1.0 else 0.0
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate tasks using the policy model"""
    
    def __init__(self, config):
        """Initialize evaluator"""
        self.config = config
    
    def evaluate_tasks(self, policy_model, tasks, api_executor=None):
        """Evaluate tasks using the policy model - WITH DETAILED RESULTS"""
        logger.info("Starting task evaluation...")
        
        results = {
            'tasks_completed': 0,
            'total_tasks': len(tasks),
            'average_score': 0.0,
            'win_rate': 0.0,
            'task_results': []  # NEW: Detailed results for each task
        }
        
        completed = 0
        total_score = 0
        
        # Print header
        logger.info("\n" + "=" * 80)
        logger.info("DETAILED EVALUATION RESULTS")
        logger.info("=" * 80)
        
        for task_idx, task in enumerate(tasks, 1):
            try:
                instruction = task.get('instruction', '')
                
                if not instruction:
                    logger.warning(f"Task {task_idx}: No instruction found, skipping")
                    continue
                
                logger.info(f"\n[Task {task_idx}/{len(tasks)}]")
                logger.info(f"Instruction: {instruction[:70]}...")
                
                # Generate response
                response = policy_model.generate(instruction)
                
                logger.info(f"Generated Response: {response[:100]}...")
                
                # Get reward using PolicyModel's method
                reward = policy_model.evaluate_response(instruction, response)
                
                logger.info(f"Score: {reward:.4f}")
                logger.info("─" * 80)
                
                # Store detailed result
                task_result = {
                    'task_number': task_idx,
                    'instruction': instruction[:100],  # Store first 100 chars
                    'full_instruction': instruction,   # Store full instruction
                    'response': response[:200],         # Store first 200 chars
                    'full_response': response,          # Store full response
                    'score': round(reward, 4)
                }
                
                results['task_results'].append(task_result)
                
                total_score += reward
                completed += 1
            
            except Exception as e:
                logger.warning(f"Task {task_idx} evaluation failed: {e}")
                continue
        
        # Calculate metrics
        if completed > 0:
            results['tasks_completed'] = completed
            results['average_score'] = round(total_score / completed, 4)
            
            # Win rate: percentage of tasks with score > 0.5
            wins = sum(1 for tr in results['task_results'] if tr['score'] > 0.5)
            results['win_rate'] = round((wins / completed) * 100, 2)
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Tasks Completed: {results['tasks_completed']}/{len(tasks)}")
        logger.info(f"Average Score: {results['average_score']:.4f}")
        logger.info(f"Win Rate: {results['win_rate']:.2f}%")
        logger.info(f"Total Tasks Evaluated: {len(results['task_results'])}")
        logger.info("=" * 80 + "\n")
        
        return results

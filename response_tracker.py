# # response_tracker_ENHANCED.py
# # ENHANCED RESPONSE TRACKER - Detailed before/after comparison
# # Matches your desired JSON format with full training comparisons

# import logging
# import json
# from pathlib import Path
# from typing import List, Dict, Any, Optional
# from datetime import datetime

# logger = logging.getLogger(__name__)


# class ResponseTracker:
#     """
#     Enhanced response tracker that captures detailed before/after comparisons
#     Format matches your example JSON structure
#     """
    
#     def __init__(self):
#         """Initialize response tracker"""
#         self.comparisons = []
#         self.start_time = datetime.now()
#         logger.info("[ResponseTracker] Initialized")
    
#     def add_comparison(
#         self,
#         task_number: int,
#         instruction: str,
#         before_response: str,
#         before_score: float,
#         after_response: str,
#         after_score: float,
#         retrieved_data: str = "",
#         full_instruction: Optional[str] = None,
#         full_before_response: Optional[str] = None,
#         full_after_response: Optional[str] = None
#     ):
#         """
#         Add a before/after training comparison
        
#         Args:
#             task_number: Task number
#             instruction: Task instruction
#             before_response: Response before training
#             before_score: Score before training
#             after_response: Response after training
#             after_score: Score after training
#             retrieved_data: Retrieved Wikipedia data
#             full_instruction: Full instruction (optional)
#             full_before_response: Full response before (optional)
#             full_after_response: Full response after (optional)
#         """
        
#         # Calculate improvement
#         score_gain = after_score - before_score
#         improvement_percent = 0.0
#         if before_score > 0:
#             improvement_percent = (score_gain / before_score) * 100
#         else:
#             improvement_percent = (score_gain * 100) if score_gain != 0 else 0.0
        
#         response_improved = after_response != before_response and after_score > before_score
        
#         # Create comparison entry
#         comparison = {
#             'task_number': task_number,
#             'instruction': instruction,
#             'full_instruction': full_instruction or instruction,
#             'before_training': {
#                 'response': before_response,
#                 'full_response': full_before_response or before_response,
#                 'score': round(before_score, 4)
#             },
#             'retrieved_data': retrieved_data[:200] if retrieved_data else "",  # First 200 chars
#             'after_training': {
#                 'response': after_response,
#                 'full_response': full_after_response or after_response,
#                 'score': round(after_score, 4)
#             },
#             'improvement': {
#                 'score_gain': round(score_gain, 4),
#                 'improvement_percent': round(improvement_percent, 2),
#                 'response_improved': response_improved
#             }
#         }
        
#         self.comparisons.append(comparison)
        
#         logger.info(
#             f"[ResponseTracker] Task {task_number}: "
#             f"Score {before_score:.4f} → {after_score:.4f} "
#             f"({improvement_percent:+.2f}%)"
#         )
    
#     def get_summary(self) -> Dict[str, Any]:
#         """Get summary statistics"""
        
#         if not self.comparisons:
#             return {
#                 'total_tasks_tracked': 0,
#                 'avg_before_score': 0.0,
#                 'avg_after_score': 0.0,
#                 'avg_improvement': 0.0,
#                 'tasks_improved': 0,
#                 'improvement_rate_percent': 0.0,
#                 'overall_improvement_percent': 0.0
#             }
        
#         before_scores = [c['before_training']['score'] for c in self.comparisons]
#         after_scores = [c['after_training']['score'] for c in self.comparisons]
#         improvements = [c['improvement']['score_gain'] for c in self.comparisons]
#         improved_tasks = [c for c in self.comparisons if c['improvement']['response_improved']]
        
#         avg_before = sum(before_scores) / len(before_scores)
#         avg_after = sum(after_scores) / len(after_scores)
#         avg_improvement = sum(improvements) / len(improvements)
#         tasks_improved = len(improved_tasks)
#         improvement_rate = (tasks_improved / len(self.comparisons)) * 100
        
#         # Overall improvement percent
#         if avg_before > 0:
#             overall_improvement_percent = ((avg_after - avg_before) / avg_before) * 100
#         else:
#             overall_improvement_percent = 0.0
        
#         return {
#             'total_tasks_tracked': len(self.comparisons),
#             'avg_before_score': round(avg_before, 4),
#             'avg_after_score': round(avg_after, 4),
#             'avg_improvement': round(avg_improvement, 4),
#             'tasks_improved': tasks_improved,
#             'improvement_rate_percent': round(improvement_rate, 2),
#             'overall_improvement_percent': round(overall_improvement_percent, 2)
#         }
    
#     def save(self, output_file: str = "results/before_after_comparison.json"):
#         """Save detailed comparison to JSON"""
        
#         output_path = Path(output_file)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
        
#         data = {
#             'total_tasks': len(self.comparisons),
#             'comparisons': self.comparisons,
#             'summary': self.get_summary(),
#             'timestamp': datetime.now().isoformat()
#         }
        
#         with open(output_path, 'w') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
        
#         logger.info(f"✓ Saved before/after comparison to {output_file}")
    
#     def print_summary(self):
#         """Print summary to console"""
        
#         summary = self.get_summary()
        
#         logger.info("\n" + "=" * 80)
#         logger.info("BEFORE/AFTER TRAINING SUMMARY")
#         logger.info("=" * 80)
#         logger.info(f"Total tasks tracked: {summary['total_tasks_tracked']}")
#         logger.info(f"Average score BEFORE: {summary['avg_before_score']:.4f}")
#         logger.info(f"Average score AFTER:  {summary['avg_after_score']:.4f}")
#         logger.info(f"Average improvement:  +{summary['avg_improvement']:.4f}")
#         logger.info(f"Tasks improved:       {summary['tasks_improved']}/{summary['total_tasks_tracked']}")
#         logger.info(f"Improvement rate:     {summary['improvement_rate_percent']:.2f}%")
#         logger.info(f"Overall improvement:  +{summary['overall_improvement_percent']:.2f}%")
#         logger.info("=" * 80)
        
#         # Print detailed comparisons
#         logger.info("\n" + "=" * 80)
#         logger.info("DETAILED COMPARISONS")
#         logger.info("=" * 80)
        
#         for comp in self.comparisons[:10]:  # Show first 10
#             logger.info(f"\nTask {comp['task_number']}: {comp['instruction'][:60]}...")
#             logger.info(f"  Before: {comp['before_training']['response'][:50]}... (Score: {comp['before_training']['score']})")
#             logger.info(f"  After:  {comp['after_training']['response'][:50]}... (Score: {comp['after_training']['score']})")
#             logger.info(f"  Improvement: {comp['improvement']['improvement_percent']:+.2f}%")
    
#     def get_comparisons(self) -> List[Dict[str, Any]]:
#         """Get all comparisons"""
#         return self.comparisons
    
#     def get_json_structure(self) -> Dict[str, Any]:
#         """Get complete JSON structure matching your format"""
#         return {
#             'total_tasks': len(self.comparisons),
#             'comparisons': self.comparisons,
#             'summary': self.get_summary()
#         }
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class ResponseTracker:
    """
    Tracks before/after responses and improvements.
    Now with:
      - Placeholder 'retrieved_data' sanitization
      - De-duplication on (task_number, instruction)
    """

    def __init__(self, save_path: str = "results/before_after_comparison.json"):
        self.save_path = save_path
        self.comparisons: List[Dict[str, Any]] = []
        self.summary: Dict[str, Any] = {}
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _sanitize_retrieved(self, retrieved_data: Optional[str]) -> str:
        if not retrieved_data:
            return ""
        lowered = retrieved_data.strip().lower()
        if lowered in {"answer", "snippets", "answer\n\nsnippets"}:
            return ""
        return retrieved_data

    def add_comparison(
        self,
        task_number: int,
        instruction: str,
        full_instruction: str,
        before_response: str,
        before_full_response: str,
        before_score: float,
        retrieved_data: str,
        after_response: str,
        after_full_response: str,
        after_score: float,
    ):
        # sanitize placeholder retrieved_data
        retrieved_data = self._sanitize_retrieved(retrieved_data)

        improvement = {
            "score_gain": round(after_score - before_score, 6),
            "improvement_percent": round(
                ((after_score - before_score) / max(1e-9, abs(before_score))) * 100.0, 2
            ) if before_score != 0 else (100.0 if after_score > 0 else 0.0),
            "response_improved": after_score > before_score
        }

        comparison = {
            "task_number": task_number,
            "instruction": instruction,
            "full_instruction": full_instruction,
            "before_training": {
                "response": before_response,
                "full_response": before_full_response,
                "score": before_score
            },
            "retrieved_data": retrieved_data,
            "after_training": {
                "response": after_response,
                "full_response": after_full_response,
                "score": after_score
            },
            "improvement": improvement
        }

        # de-duplicate on (task_number, instruction): replace if exists
        replaced = False
        for i, c in enumerate(self.comparisons):
            if c.get('task_number') == task_number and c.get('instruction') == instruction:
                self.comparisons[i] = comparison
                replaced = True
                break
        if not replaced:
            self.comparisons.append(comparison)

    def finalize(self):
        total = len(self.comparisons)
        if total == 0:
            self.summary = {
                "total_tasks_tracked": 0,
                "avg_before_score": 0.0,
                "avg_after_score": 0.0,
                "avg_improvement": 0.0,
                "tasks_improved": 0,
                "improvement_rate_percent": 0.0,
                "overall_improvement_percent": 0.0,
            }
        else:
            avg_before = sum(c["before_training"]["score"] for c in self.comparisons) / total
            avg_after = sum(c["after_training"]["score"] for c in self.comparisons) / total
            avg_improve = (avg_after - avg_before)
            improved = sum(1 for c in self.comparisons if c["improvement"]["response_improved"])
            improvement_rate = (improved / total) * 100.0
            overall_improvement_percent = (avg_improve / max(1e-9, abs(avg_before))) * 100.0 if avg_before != 0 else (100.0 if avg_after > 0 else 0.0)

            self.summary = {
                "total_tasks_tracked": total,
                "avg_before_score": round(avg_before, 6),
                "avg_after_score": round(avg_after, 6),
                "avg_improvement": round(avg_improve, 6),
                "tasks_improved": improved,
                "improvement_rate_percent": round(improvement_rate, 2),
                "overall_improvement_percent": round(overall_improvement_percent, 2),
            }

    def save(self):
        self.finalize()
        payload = {
            "total_tasks": len(self.comparisons),
            "comparisons": self.comparisons,
            "summary": self.summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return self.save_path
    
    def print_summary(self):
        """Backwards-compatible: prints summary if already computed, else finalizes it."""
        if not self.summary:
            self.finalize()

        print("\n===== RESPONSE TRACKER SUMMARY =====")
        print(json.dumps(self.summary, indent=2))
        print("====================================\n")


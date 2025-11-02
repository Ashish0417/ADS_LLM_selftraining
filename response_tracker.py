import logging
import json
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class ResponseTracker:
    """Track responses before and after training"""
    
    def __init__(self, output_file: str = "results/before_after_comparison.json"):
        self.output_file = output_file
        self.comparisons = []
    
    def add_comparison(self, 
                       task_number: int,
                       instruction: str,
                       before_response: str,
                       before_score: float,
                       after_response: str,
                       after_score: float,
                       retrieved_data: str = ""):
        """Add before/after comparison for a task"""
        
        comparison = {
            'task_number': task_number,
            'instruction': instruction[:100],
            'full_instruction': instruction,
            
            'before_training': {
                'response': before_response[:150],
                'full_response': before_response,
                'score': round(before_score, 4)
            },
            
            'retrieved_data': retrieved_data[:200] if retrieved_data else "",
            
            'after_training': {
                'response': after_response[:150],
                'full_response': after_response,
                'score': round(after_score, 4)
            },
            
            'improvement': {
                'score_gain': round(after_score - before_score, 4),
                'improvement_percent': round(((after_score - before_score) / max(before_score, 0.01)) * 100, 2),
                'response_improved': after_score > before_score
            }
        }
        
        self.comparisons.append(comparison)
        logger.info(f"  │ [TRACKED] Task {task_number}: Score {comparison['before_training']['score']:.4f} → {comparison['after_training']['score']:.4f}")
    
    def save(self):
        """Save comparisons to JSON"""
        with open(self.output_file, 'w') as f:
            json.dump({
                'total_tasks': len(self.comparisons),
                'comparisons': self.comparisons,
                'summary': self._get_summary()
            }, f, indent=2)
        
        logger.info(f"\n✓ Saved before/after comparison to {self.output_file}")
    
    def _get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.comparisons:
            return {}
        
        improvements = [c['improvement']['score_gain'] for c in self.comparisons]
        improvements_positive = sum(1 for imp in improvements if imp > 0)
        
        avg_before = sum(c['before_training']['score'] for c in self.comparisons) / len(self.comparisons)
        avg_after = sum(c['after_training']['score'] for c in self.comparisons) / len(self.comparisons)
        
        return {
            'total_tasks_tracked': len(self.comparisons),
            'avg_before_score': round(avg_before, 4),
            'avg_after_score': round(avg_after, 4),
            'avg_improvement': round(sum(improvements) / len(self.comparisons), 4),
            'tasks_improved': improvements_positive,
            'improvement_rate_percent': round((improvements_positive / len(self.comparisons)) * 100, 2),
            'overall_improvement_percent': round(((avg_after - avg_before) / max(avg_before, 0.01)) * 100, 2)
        }
    
    def print_summary(self):
        """Print summary to console"""
        summary = self._get_summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("BEFORE/AFTER TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total tasks tracked: {summary.get('total_tasks_tracked', 0)}")
        logger.info(f"Average score BEFORE: {summary.get('avg_before_score', 0):.4f}")
        logger.info(f"Average score AFTER:  {summary.get('avg_after_score', 0):.4f}")
        logger.info(f"Average improvement:  {summary.get('avg_improvement', 0):+.4f}")
        logger.info(f"Tasks improved:       {summary.get('tasks_improved', 0)}/{summary.get('total_tasks_tracked', 0)}")
        logger.info(f"Improvement rate:     {summary.get('improvement_rate_percent', 0):.2f}%")
        logger.info(f"Overall improvement:  {summary.get('overall_improvement_percent', 0):+.2f}%")
        logger.info("=" * 80)

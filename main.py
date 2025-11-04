# main_WITH_ENHANCED_TRACKING.py
# UPDATED MAIN - Uses Enhanced Response Tracking
# Produces detailed before/after JSON like your example

import logging
import os
import sys
import json
import torch
from typing import List, Dict, Any
from pathlib import Path

# Import components
from config import ADSConfig
from utils import DataCache, MetricsLogger, ProgressTracker, save_json, load_json
from data_loader import DataManager  # ← Use Dolly loader
from policy_model import ModelManager
from retrieval_engine import AdvancedRetrievalEngine
from evaluator import Evaluator
from response_tracker import ResponseTracker  # ← Use enhanced tracker

logger = logging.getLogger(__name__)


class APIExecutor:
    """Execute API with Advanced Retrieval Engine"""
    
    def __init__(self, retrieval_engine: AdvancedRetrievalEngine):
        """Initialize executor with retrieval engine"""
        self.retrieval_engine = retrieval_engine
        logger.info("APIExecutor initialized with Advanced Retrieval Engine")
    
    def execute_trajectory(self, trajectory, instruction: str = ""):
        """Execute trajectory using advanced retrieval"""
        
        if not instruction or not instruction.strip():
            return {'collected_data': '', 'total_cost': 0, 'api_calls': []}
        
        try:
            # Use advanced retrieval engine
            results = self.retrieval_engine.retrieve(instruction, top_k=3)
            
            if results:
                collected_data = "\n\n".join(results)
            else:
                collected_data = ""
            
            return {
                'collected_data': collected_data,
                'total_cost': 1,
                'api_calls': ['advanced_retrieval']
            }
        
        except Exception as e:
            logger.error(f"[APIExecutor] Error: {e}")
            return {'collected_data': '', 'total_cost': 0, 'api_calls': []}


class ADSFrameworkWithEnhancedTracking:
    """
    ADS Framework with Enhanced Before/After Tracking
    Produces detailed JSON comparisons for each task
    """
    
    def __init__(self, config: ADSConfig = None):
        """Initialize ADS framework"""
        if config is None:
            config = ADSConfig
        
        self.config = config
        self.data_manager = None
        self.model_manager = None
        self.retrieval_engine = None
        self.api_executor = None
        self.evaluator = None
        self.metrics_logger = None
        self.response_tracker = None
        
        logger.info("=" * 80)
        logger.info("INITIALIZING ADS FRAMEWORK - WITH ENHANCED TRACKING")
        logger.info("=" * 80)
        config.print_config()
    
    def setup(self):
        """Setup all components"""
        logger.info("\n[STEP 1/4] Loading Data...")
        self._setup_data()
        
        logger.info("\n[STEP 2/4] Initializing Models...")
        self._setup_models()
        
        logger.info("\n[STEP 3/4] Initializing Advanced Retrieval Engine...")
        self._setup_retrieval_engine()
        
        logger.info("\n[STEP 4/4] Initializing Evaluator...")
        self._setup_evaluator()
        
        logger.info("\n" + "=" * 80)
        logger.info("SETUP COMPLETE - READY FOR TRAINING")
        logger.info("=" * 80 + "\n")
    
    def _setup_data(self):
        """Setup data with Dolly tasks"""
        config_dict = {
            'dataset_config': self.config.DATASET_CONFIG,
            'wiki_docs': 0,
            'magpie_tasks': self.config.DATASET_CONFIG.get('total_tasks', 100),
        }
        
        self.data_manager = DataManager(config_dict)
        self.data_manager.prepare_all_data()
        self.data_loaders = self.data_manager.get_data_loaders()
        
        logger.info("✓ Data loaded (Dolly tasks)")
    
    def _setup_models(self):
        """Setup LLM models"""
        config_dict = {
            'policy_model_name': self.config.POLICY_MODEL_NAME,
            'device': self.config.DEVICE,
            'load_in_8bit': self.config.LOAD_IN_8BIT,
        }
        self.model_manager = ModelManager(config_dict)
        self.model_manager.initialize_models()
        
        self.policy_model = self.model_manager.get_policy_model()
        self.optimizer_model = self.model_manager.get_optimizer_model()
        
        logger.info("✓ Models initialized")
    
    def _setup_retrieval_engine(self):
        """Setup Advanced Retrieval Engine"""
        self.retrieval_engine = AdvancedRetrievalEngine(
            use_cross_encoder=True,
            use_dense=True
        )
        
        self.api_executor = APIExecutor(self.retrieval_engine)
        logger.info("✓ Advanced Retrieval Engine initialized")
    
    def _setup_evaluator(self):
        """Setup evaluator"""
        self.evaluator = Evaluator(self.config)
        logger.info("✓ Evaluator initialized")
    
    def train(self, num_iterations: int = None):
        """Main training loop with detailed tracking"""
        if num_iterations is None:
            num_iterations = self.config.TRAINING_CONFIG['num_iterations']
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING - WITH ENHANCED TRACKING")
        logger.info("=" * 80 + "\n")
        
        # Initialize trackers
        log_file = os.path.join(self.config.RESULTS_DIR, "training_metrics.json")
        self.metrics_logger = MetricsLogger(log_file)
        self.response_tracker = ResponseTracker()  # ← Enhanced tracker
        
        train_tasks = self.data_loaders['train'][:5]  # Use subset
        
        for iteration in range(num_iterations):
            logger.info(f"\n{'='*80}")
            logger.info(f"ITERATION {iteration + 1}/{num_iterations}")
            logger.info(f"{'='*80}\n")
            
            iteration_metrics = {
                'iteration': iteration + 1,
                'num_tasks': len(train_tasks),
                'total_api_cost': 0,
                'avg_reward': 0,
                'completed_tasks': 0,
            }
            
            progress = ProgressTracker(len(train_tasks), f"Processing tasks (iter {iteration+1})")
            
            for task_idx, task in enumerate(train_tasks):
                try:
                    instruction = task.get('instruction', '')
                    context = task.get('context', '')
                    
                    # Combine instruction and context
                    full_instruction = f"{instruction}\n{context}" if context else instruction
                    
                    logger.info(f"\n  Task {task_idx + 1}/{len(train_tasks)}: {instruction[:50]}...")
                    
                    # ============ CAPTURE BEFORE TRAINING ============
                    before_response = self.policy_model.generate(instruction)
                    before_score = self.policy_model.evaluate_response(instruction, before_response)
                    
                    logger.info(f"\n  ┌─ [BEFORE Training]")
                    logger.info(f"  │ Response: {before_response[:70]}...")
                    logger.info(f"  │ Score: {before_score:.4f}")
                    logger.info(f"  └─")
                    
                    # ============ ADVANCED RETRIEVAL ============
                    logger.info("\n  ┌─" + "─" * 76 + "┐")
                    logger.info("  │ ADVANCED RETRIEVAL ENGINE")
                    logger.info("  │ (Entity Linking → BM25 → Dense → Cross-Encoder)")
                    logger.info("  ├─" + "─" * 76 + "┤")
                    logger.info(f"  │ Query: {instruction[:70]}...")
                    
                    try:
                        api_results = self.api_executor.execute_trajectory(
                            trajectory=None,
                            instruction=instruction
                        )
                    except Exception as e:
                        logger.error(f"Retrieval error: {e}")
                        api_results = {'collected_data': ''}
                    
                    collected_data = api_results.get('collected_data', '')
                    
                    logger.info(f"  │")
                    logger.info(f"  │ RETRIEVED DATA (Advanced Retrieval):")
                    logger.info(f"  │ {'-' * 74}")
                    
                    if collected_data:
                        display_text = collected_data[:500]
                        if len(collected_data) > 500:
                            display_text += "..."
                        
                        for line in display_text.split('\n')[:5]:
                            if line.strip():
                                wrapped_line = line[:72] if len(line) <= 72 else line[:69] + "..."
                                logger.info(f"  │ {wrapped_line}")
                    else:
                        logger.info(f"  │ [No data retrieved]")
                    
                    logger.info(f"  │ {'-' * 74}")
                    logger.info("  │ [Updating policy with in-context learning...]")
                    logger.info("  └─" + "─" * 76 + "┘\n")
                    
                    # ============ CAPTURE AFTER TRAINING ============
                    if collected_data:
                        after_response = self.policy_model.in_context_learn(
                            instruction=instruction,
                            examples=[{'input': collected_data}]
                        )
                        after_score = self.policy_model.evaluate_response(instruction, after_response)
                        
                        logger.info(f"  ┌─ [AFTER Training]")
                        logger.info(f"  │ Response: {after_response[:70]}...")
                        logger.info(f"  │ Score: {after_score:.4f}")
                        
                        improvement = after_score - before_score
                        improvement_percent = (improvement / max(before_score, 0.01)) * 100
                        
                        improvement_symbol = "⬆️" if improvement > 0 else "⬇️" if improvement < 0 else "→"
                        logger.info(f"  │ {improvement_symbol} Improvement: {improvement:+.4f} ({improvement_percent:+.2f}%)")
                        logger.info(f"  └─")
                    else:
                        after_response = before_response
                        after_score = before_score
                        logger.info(f"  ┌─ [AFTER Training - Baseline]")
                        logger.info(f"  │ (No retrieved data)")
                        logger.info(f"  └─")
                    
                    # ============ TRACK COMPARISON (Enhanced format) ============
                    self.response_tracker.add_comparison(
                        task_number=task_idx + 1,
                        instruction=instruction,
                        before_response=before_response,
                        before_score=before_score,
                        after_response=after_response,
                        after_score=after_score,
                        retrieved_data=collected_data,
                        full_instruction=full_instruction,
                        full_before_response=before_response,
                        full_after_response=after_response
                    )
                    
                    iteration_metrics['total_api_cost'] += api_results.get('total_cost', 0)
                    iteration_metrics['avg_reward'] += after_score
                    iteration_metrics['completed_tasks'] += 1
                    
                    progress.update(1)
                
                except Exception as e:
                    logger.error(f"Task processing failed: {e}")
                    progress.update(1)
            
            progress.finish()
            
            # Log iteration metrics
            if iteration_metrics['completed_tasks'] > 0:
                iteration_metrics['avg_reward'] /= iteration_metrics['completed_tasks']
            
            self.metrics_logger.log(**iteration_metrics)
            
            logger.info(f"\n  Iteration Summary:")
            logger.info(f"    - Tasks completed: {iteration_metrics['completed_tasks']}/{len(train_tasks)}")
            logger.info(f"    - Average reward: {iteration_metrics['avg_reward']:.4f}")
            logger.info(f"    - Total API cost: {iteration_metrics['total_api_cost']}")
        
        # Save results with enhanced format
        self.metrics_logger.save()
        self.response_tracker.save()
        self.response_tracker.print_summary()
        
        # Print retrieval statistics
        stats = self.retrieval_engine.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("RETRIEVAL ENGINE STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total queries: {stats['total_queries']}")
        logger.info(f"Entity-linked results: {stats['entity_linked']}")
        if stats['total_queries'] > 0:
            logger.info(f"Entity link rate: {(stats['entity_linked'] / stats['total_queries']) * 100:.1f}%")
            logger.info(f"Average retrieval time: {stats['avg_time']:.2f}s")
        logger.info("=" * 80)
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
    
    def evaluate(self):
        """Evaluate on test set"""
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("=" * 80 + "\n")
        
        test_tasks = self.data_loaders['test'][:3]
        
        results = self.evaluator.evaluate_tasks(
            policy_model=self.policy_model,
            tasks=test_tasks,
            api_executor=self.api_executor
        )
        
        results_file = os.path.join(self.config.RESULTS_DIR, "evaluation_results.json")
        save_json(results, results_file)
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
        return results
    
    def save_checkpoint(self, path: str = None):
        """Save checkpoint"""
        if path is None:
            path = os.path.join(self.config.RESULTS_DIR, "checkpoint.pt")
        
        try:
            checkpoint = {
                'model_name': self.policy_model.model_name,
                'model_state': self.policy_model.model.state_dict() if hasattr(self.policy_model.model, 'state_dict') else None,
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING FULL ADS PIPELINE - WITH ENHANCED TRACKING")
        logger.info("=" * 80 + "\n")
        
        self.setup()
        self.train(num_iterations=self.config.TRAINING_CONFIG['num_iterations'])
        results = self.evaluate()
        self.save_checkpoint()
        
        logger.info("\n" + "=" * 80)
        logger.info("FULL PIPELINE COMPLETE!")
        logger.info("=" * 80)
        
        return results


def main():
    """Main entry point"""
    ads = ADSFrameworkWithEnhancedTracking(ADSConfig)
    results = ads.run_full_pipeline()
    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ads_framework.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        results = main()
        logger.info("SUCCESS: ADS Framework execution completed")
        logger.info(f"Results saved to results/before_after_comparison.json")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

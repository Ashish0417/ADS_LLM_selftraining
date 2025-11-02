import logging
import os
import json
import torch
from typing import List, Dict, Any
from pathlib import Path

# Import all components
from config import ADSConfig
from utils import DataCache, MetricsLogger, ProgressTracker, save_json, load_json
from data_loader import DataManager
from policy_model import ModelManager
from api_handler import (
    HybridRetrievalAPI, DemonstrationGenerationAPI, QuestionAnsweringAPI, APIExecutor
)
from evaluator import Evaluator
from response_tracker import ResponseTracker

logger = logging.getLogger(__name__)


class ADSFramework:
    """Active Data Search Framework with Hybrid Retrieval"""
    
    def __init__(self, config: ADSConfig = None):
        """Initialize ADS framework"""
        if config is None:
            config = ADSConfig
        
        self.config = config
        self.data_manager = None
        self.model_manager = None
        self.api_executor = None
        self.evaluator = None
        self.metrics_logger = None
        self.response_tracker = None
        self.training_data_log = []
        
        logger.info("=" * 80)
        logger.info("INITIALIZING ADS FRAMEWORK WITH HYBRID RETRIEVAL")
        logger.info("=" * 80)
        config.print_config()
    
    def setup(self):
        """Setup all components"""
        logger.info("\n[STEP 1/4] Loading Data...")
        self._setup_data()
        
        logger.info("\n[STEP 2/4] Initializing Models...")
        self._setup_models()
        
        logger.info("\n[STEP 3/4] Initializing APIs (Hybrid Retrieval)...")
        self._setup_apis()
        
        logger.info("\n[STEP 4/4] Initializing Evaluator...")
        self._setup_evaluator()
        
        logger.info("\n" + "=" * 80)
        logger.info("SETUP COMPLETE - READY FOR TRAINING")
        logger.info("=" * 80 + "\n")
    
    def _setup_data(self):
        """Setup data"""
        config_dict = {
            'dataset_config': self.config.DATASET_CONFIG,
            'wiki_docs': self.config.WIKIPEDIA_CONFIG.get('cache_size', 5000),
            'magpie_tasks': self.config.DATASET_CONFIG.get('total_tasks', 100),
        }
        
        self.data_manager = DataManager(config_dict)
        self.data_manager.prepare_all_data()
        self.data_loaders = self.data_manager.get_data_loaders()
    
    def _setup_models(self):
        """Setup models"""
        config_dict = {
            'policy_model_name': self.config.POLICY_MODEL_NAME,
            'device': self.config.DEVICE,
            'load_in_8bit': self.config.LOAD_IN_8BIT,
        }
        self.model_manager = ModelManager(config_dict)
        self.model_manager.initialize_models()
        
        self.policy_model = self.model_manager.get_policy_model()
        self.optimizer_model = self.model_manager.get_optimizer_model()
    
    def _setup_apis(self):
        """Setup APIs with Hybrid Retrieval"""
        wikipedia_docs = self.data_loaders['wikipedia']
        
        # Initialize Hybrid Retrieval API (BM25 + Semantic + Online Fallback)
        ir_api = HybridRetrievalAPI(
            documents=wikipedia_docs,
            cache_dir=self.config.CACHE_DIR,
            use_semantic=True  # Enable semantic search
        )
        
        # Save reference for use in training
        self.ir_api = ir_api
        
        demo_api = DemonstrationGenerationAPI(
            generator_model=self.policy_model.model,
            tokenizer=self.policy_model.tokenizer,
            cache_dir=self.config.CACHE_DIR
        )
        
        qa_api = QuestionAnsweringAPI(
            qa_model=self.policy_model.model,
            tokenizer=self.policy_model.tokenizer,
            cache_dir=self.config.CACHE_DIR
        )
        
        # Create executor
        self.api_executor = APIExecutor(ir_api, demo_api, qa_api)
        logger.info("✓ APIs initialized with Hybrid Retrieval")
    
    def _setup_evaluator(self):
        """Setup evaluator"""
        self.evaluator = Evaluator(self.config)
    
    def train(self, num_iterations: int = None):
        """Main training loop with Hybrid Retrieval"""
        if num_iterations is None:
            num_iterations = self.config.TRAINING_CONFIG['num_iterations']
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING WITH HYBRID RETRIEVAL")
        logger.info("=" * 80 + "\n")
        
        # Initialize trackers
        log_file = os.path.join(self.config.RESULTS_DIR, "training_metrics.json")
        self.metrics_logger = MetricsLogger(log_file)
        self.response_tracker = ResponseTracker()
        
        train_tasks = self.data_loaders['train'][:5]  # Use subset for speed
        
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
                    logger.info(f"\n  Task {task_idx + 1}/{len(train_tasks)}: {instruction[:50]}...")
                    
                    # ============ CAPTURE BEFORE TRAINING ============
                    before_response = self.policy_model.generate(instruction)
                    before_score = self.policy_model.evaluate_response(instruction, before_response)
                    
                    logger.info(f"\n  ┌─ [BEFORE Training]")
                    logger.info(f"  │ Response: {before_response[:70]}...")
                    logger.info(f"  │ Score: {before_score:.4f}")
                    logger.info(f"  └─")
                    
                    # ============ HYBRID RETRIEVAL (BM25 + Semantic + Online) ============
                    logger.info("\n  ┌─" + "─" * 76 + "┐")
                    logger.info("  │ HYBRID RETRIEVAL (BM25 + Semantic + Online Wikipedia)")
                    logger.info("  ├─" + "─" * 76 + "┤")
                    logger.info(f"  │ Instruction: {instruction[:70]}...")
                    
                    try:
                        # Use Hybrid Retrieval directly (tries local, then online)
                        retrieved = self.ir_api.retrieve(instruction, top_k=5)
                        collected_data = "\n\n".join(retrieved) if retrieved else ""
                        
                        api_results = {
                            'collected_data': collected_data,
                            'total_cost': 1,
                            'api_calls': ['hybrid_retrieval']
                        }
                    except Exception as e:
                        logger.error(f"Retrieval error: {e}")
                        api_results = {'collected_data': ''}
                    
                    # Print retrieved data
                    logger.info(f"  │")
                    logger.info(f"  │ RETRIEVED DATA (Hybrid Method):")
                    logger.info(f"  │ {'-' * 74}")
                    
                    if collected_data:
                        # Show first 500 chars of retrieved data
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
                    logger.info("  │")
                    logger.info("  │ [Updating policy with in-context learning...]")
                    logger.info("  └─" + "─" * 76 + "┘\n")
                    
                    # ============ CAPTURE AFTER TRAINING ============
                    if api_results and api_results.get('collected_data'):
                        after_response = self.policy_model.in_context_learn(
                            instruction=instruction,
                            examples=[{'input': api_results['collected_data']}]
                        )
                        after_score = self.policy_model.evaluate_response(instruction, after_response)
                        
                        logger.info(f"  ┌─ [AFTER Training]")
                        logger.info(f"  │ Response: {after_response[:70]}...")
                        logger.info(f"  │ Score: {after_score:.4f}")
                        
                        # Calculate improvement
                        improvement = after_score - before_score
                        improvement_percent = (improvement / max(before_score, 0.01)) * 100
                        
                        improvement_symbol = "⬆️" if improvement > 0 else "⬇️" if improvement < 0 else "→"
                        logger.info(f"  │ {improvement_symbol} Improvement: {improvement:+.4f} ({improvement_percent:+.1f}%)")
                        logger.info(f"  └─")
                        
                        # Track comparison
                        self.response_tracker.add_comparison(
                            task_number=task_idx + 1,
                            instruction=instruction,
                            before_response=before_response,
                            before_score=before_score,
                            after_response=after_response,
                            after_score=after_score,
                            retrieved_data=collected_data
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
        
        # Save metrics and comparisons
        self.metrics_logger.save()
        self.response_tracker.save()
        self.response_tracker.print_summary()
        
        metrics_summary = self.metrics_logger.summary()
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Summary:")
        for key, value in metrics_summary.items():
            logger.info(f"  {key}: {value}")
    
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
                'iteration': getattr(self, 'current_iteration', 0),
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def run_full_pipeline(self):
        """Run complete ADS pipeline with Hybrid Retrieval"""
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING FULL ADS PIPELINE WITH HYBRID RETRIEVAL")
        logger.info("=" * 80 + "\n")
        
        # Setup
        self.setup()
        
        # Train
        self.train(num_iterations=self.config.TRAINING_CONFIG['num_iterations'])
        
        # Evaluate
        results = self.evaluate()
        
        # Save checkpoint
        self.save_checkpoint()
        
        logger.info("\n" + "=" * 80)
        logger.info("FULL PIPELINE COMPLETE!")
        logger.info("=" * 80)
        
        return results


def main():
    """Main entry point"""
    ads = ADSFramework(ADSConfig)
    results = ads.run_full_pipeline()
    return results


if __name__ == "__main__":
    import sys
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
        logger.info(f"\nResults saved to results/")
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)


# import logging
# import os
# import sys
# import json
# import torch
# from typing import List, Dict, Any
# from pathlib import Path
# import wikipedia

# from config import ADSConfig
# from utils import DataCache, MetricsLogger, ProgressTracker, save_json, load_json
# from data_loader import DataManager
# from policy_model import ModelManager
# from retrieval_engine import AdvancedRetrievalEngine
# from evaluator import Evaluator
# from response_tracker import ResponseTracker
# from retrieval_engine import QAPairGenerator, QAPairStorage

# logger = logging.getLogger(__name__)


# class WikipediaDocumentLoader:
#     """Load Wikipedia documents for real-time retrieval"""
    
#     def __init__(self, cache_dir: str = "cache/wikipedia"):
#         """Initialize document loader"""
#         self.cache_dir = Path(cache_dir)
#         self.cache_dir.mkdir(parents=True, exist_ok=True)
#         self.cache_file = self.cache_dir / "wikipedia_docs.json"
#         logger.info(f"[WikipediaLoader] Initialized")
    
#     def load_documents(self, topics: List[str] = None, use_cache: bool = False) -> List[str]:
#         """
#         Load Wikipedia documents
        
#         Args:
#             topics: Specific topics to load
#             use_cache: Whether to use cached documents (default: False for real-time)
        
#         Returns:
#             List of Wikipedia documents
#         """
#         if topics is None:
#             topics = self._get_default_topics()
        
#         # Try cache first if enabled
#         if use_cache and self.cache_file.exists():
#             logger.info(f"[WikipediaLoader] Loading from cache...")
#             cached_docs = self._load_from_cache()
#             if cached_docs:
#                 return cached_docs
        
#         # Fetch fresh documents
#         logger.info(f"[WikipediaLoader] Fetching {len(topics)} topics from Wikipedia...")
#         documents = self._fetch_from_wikipedia(topics)
        
#         # Don't save to cache by default (real-time retrieval)
#         if use_cache and documents:
#             self._save_to_cache(documents)
        
#         return documents
    
#     def _get_default_topics(self) -> List[str]:
#         """Get default Wikipedia topics"""
#         return [
#             "Prime Minister of the United Kingdom",
#             "Keir Starmer",
#             "Academy Award for Best Picture",
#             "Formula One World Championship",
#             "iPhone",
#             "Apple Inc.",
#             "United Kingdom",
#             "2025 Formula One season",
#         ]
    
#     def _fetch_from_wikipedia(self, topics: List[str]) -> List[str]:
#         """Fetch documents from Wikipedia"""
#         documents = []
        
#         for topic in topics:
#             try:
#                 results = wikipedia.search(topic, results=2)
                
#                 if results:
#                     try:
#                         page = wikipedia.page(results[0])
#                         doc_text = page.content
                        
#                         if len(doc_text) > 100:
#                             documents.append(doc_text)
#                             logger.debug(f"[WikipediaLoader] ✓ {results[0]}")
                    
#                     except (wikipedia.exceptions.DisambiguationError, 
#                            wikipedia.exceptions.PageError):
#                         pass
            
#             except Exception as e:
#                 logger.debug(f"[WikipediaLoader] Error fetching {topic}")
        
#         logger.info(f"[WikipediaLoader] ✓ Fetched {len(documents)} documents")
#         return documents
    
#     def _save_to_cache(self, documents: List[str]):
#         """Save documents to cache"""
#         try:
#             cache_data = {
#                 'timestamp': str(__import__('datetime').datetime.now()),
#                 'num_documents': len(documents),
#                 'documents': documents
#             }
            
#             with open(self.cache_file, 'w', encoding='utf-8') as f:
#                 json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
#             logger.debug(f"[WikipediaLoader] Cached {len(documents)} docs")
        
#         except Exception as e:
#             logger.warning(f"[WikipediaLoader] Error saving cache: {e}")
    
#     def _load_from_cache(self) -> List[str]:
#         """Load documents from cache"""
#         try:
#             with open(self.cache_file, 'r', encoding='utf-8') as f:
#                 cache_data = json.load(f)
            
#             documents = cache_data.get('documents', [])
#             logger.debug(f"[WikipediaLoader] Loaded {len(documents)} from cache")
            
#             return documents
        
#         except Exception as e:
#             logger.debug(f"[WikipediaLoader] Error loading cache")
#             return []


# class ImprovedAPIExecutor:
#     """Execute API with Advanced Retrieval Engine"""
    
#     def __init__(self, retrieval_engine: AdvancedRetrievalEngine):
#         """Initialize executor"""
#         if retrieval_engine is None:
#             raise ValueError("retrieval_engine cannot be None")
        
#         self.retrieval_engine = retrieval_engine
#         logger.info("ImprovedAPIExecutor initialized")
    
#     def execute_trajectory(self, trajectory, instruction: str = ""):
#         """Execute trajectory using advanced retrieval"""
        
#         if not instruction or not instruction.strip():
#             return {'collected_data': '', 'total_cost': 0, 'api_calls': [], 'debug': 'Empty'}
        
#         try:
#             results = self.retrieval_engine.retrieve(instruction, top_k=3)
            
#             if results:
#                 collected_data = "\n\n".join(results)
#             else:
#                 collected_data = ""
            
#             return {
#                 'collected_data': collected_data,
#                 'total_cost': 1,
#                 'api_calls': ['advanced_retrieval'],
#                 'debug': f'Retrieved {len(results)} docs' if results else 'No results'
#             }
        
#         except Exception as e:
#             logger.error(f"[APIExecutor] Error: {e}")
#             return {'collected_data': '', 'total_cost': 0, 'api_calls': [], 'debug': f'Error: {str(e)}'}


# class ImprovedScoringFunction:
#     """Improved answer scoring"""
    
#     @staticmethod
#     def score_response(instruction: str, response: str) -> float:
#         """Score response"""
        
#         if not response or len(response.strip()) == 0:
#             return 0.0
        
#         response_lower = response.lower().strip()
        
#         fail_patterns = ['don\'t know', 'unclear', 'sorry', 'unable', 'unanswerable']
#         if any(p in response_lower for p in fail_patterns):
#             return 0.2
        
#         words = response.split()
#         word_count = len(words)
#         has_capitals = any(c.isupper() for c in response)
        
#         if word_count == 1:
#             return 0.7 if has_capitals else 0.4
#         elif 2 <= word_count <= 6:
#             return 0.8 if has_capitals else 0.5
#         elif 7 <= word_count <= 15:
#             return 0.6
#         elif word_count > 20:
#             return 0.4
#         else:
#             return 0.5


# class ImprovedADSFramework:
#     """
#     ADS Framework - FULLY CORRECTED VERSION
    
#     Fixes:
#     1. api_executor properly initialized
#     2. full_instruction properly defined
#     3. Real-time document loading
#     4. No persistent storage by default
#     5. All error checking in place
#     """
    
#     def __init__(self, config: ADSConfig = None):
#         """Initialize framework"""
#         if config is None:
#             config = ADSConfig
        
#         self.config = config
#         self.data_manager = None
#         self.model_manager = None
#         self.retrieval_engine = None
#         self.api_executor = None  # FIX #1: Initialize as None
#         self.evaluator = None
#         self.metrics_logger = None
#         self.response_tracker = None
#         self.qa_generator = None
#         self.qa_storage = None
        
#         logger.info("=" * 80)
#         logger.info("INITIALIZING ADS FRAMEWORK - FULLY CORRECTED")
#         logger.info("=" * 80)
#         config.print_config()
    
#     def setup(self):
#         """Setup all components"""
#         try:
#             logger.info("\n[STEP 1/6] Loading Data...")
#             self._setup_data()
            
#             logger.info("\n[STEP 2/6] Initializing Models...")
#             self._setup_models()
            
#             logger.info("\n[STEP 3/6] Initializing Retrieval Engine...")
#             self._setup_retrieval_engine()  # FIX #1: Includes APIExecutor
            
#             logger.info("\n[STEP 4/6] LOADING WIKIPEDIA DOCUMENTS...")
#             self._load_and_index_documents()
            
#             logger.info("\n[STEP 5/6] Initializing Q-A Pair Generator...")
#             self._setup_qa_generator()
            
#             logger.info("\n[STEP 6/6] Initializing Evaluator...")
#             self._setup_evaluator()
            
#             logger.info("\n" + "=" * 80)
#             logger.info("SETUP COMPLETE")
#             logger.info("=" * 80 + "\n")
        
#         except Exception as e:
#             logger.error(f"Setup failed: {e}", exc_info=True)
#             raise
    
#     def _setup_data(self):
#         """Setup data"""
#         config_dict = {
#             'dataset_config': self.config.DATASET_CONFIG,
#             'wiki_docs': 0,
#             'magpie_tasks': self.config.DATASET_CONFIG.get('total_tasks', 100),
#         }
        
#         self.data_manager = DataManager(config_dict)
#         self.data_manager.prepare_all_data()
#         self.data_loaders = self.data_manager.get_data_loaders()
#         self.data_manager.print_task_summary()
        
#         train_tasks = self.data_loaders.get('train', [])
#         logger.info(f"✓ Data loaded: {len(train_tasks)} train")
    
#     def _setup_models(self):
#         """Setup models"""
#         config_dict = {
#             'policy_model_name': self.config.POLICY_MODEL_NAME,
#             'device': self.config.DEVICE,
#             'load_in_8bit': self.config.LOAD_IN_8BIT,
#         }
#         self.model_manager = ModelManager(config_dict)
#         self.model_manager.initialize_models()
        
#         self.policy_model = self.model_manager.get_policy_model()
#         logger.info("✓ Models initialized")
    
#     def _setup_retrieval_engine(self):
#         """Setup retrieval engine - FIX #1: Added APIExecutor initialization"""
#         self.retrieval_engine = AdvancedRetrievalEngine(
#             # use_cross_encoder=True,
#             # use_dense=True
#         )
        
#         # FIX #1: INITIALIZE APIExecutor (was missing!)
#         self.api_executor = ImprovedAPIExecutor(self.retrieval_engine)
        
#         logger.info("✓ Retrieval Engine + APIExecutor initialized")
    
#     def _load_and_index_documents(self):
#         """Load and index Wikipedia documents"""
#         logger.info("\n" + "=" * 80)
#         logger.info("LOADING WIKIPEDIA DOCUMENTS")
#         logger.info("=" * 80)
        
#         # Initialize loader
#         doc_loader = WikipediaDocumentLoader()
        
#         # Load documents (use_cache=False for real-time retrieval)
#         wikipedia_documents = doc_loader.load_documents(use_cache=False)
        
#         if not wikipedia_documents:
#             logger.warning("⚠️ No Wikipedia documents loaded, using fallback...")
#             wikipedia_documents = self._create_fallback_documents()
        
#         # Index documents
#         logger.info(f"[INDEXING] Indexing {len(wikipedia_documents)} documents...")
#         self.retrieval_engine.index_documents(wikipedia_documents)
#         logger.info(f"✓ Successfully indexed {len(wikipedia_documents)} documents!")
        
#         logger.info("=" * 80)
    
#     def _create_fallback_documents(self) -> List[str]:
#         """Fallback documents if Wikipedia unavailable"""
        
#         return [
#             """Keir Starmer became Prime Minister of the United Kingdom on 5 July 2024.""",
#             """Academy Award for Best Picture is the most prestigious award in the film industry.""",
#             """Formula One World Championship is the highest class of international auto racing.""",
#             """Apple Inc. produces the iPhone, a series of smartphones.""",
#             """United Kingdom comprises England, Scotland, Wales and Northern Ireland.""",
#         ]
    
#     def _setup_qa_generator(self):
#         """Setup Q-A generator"""
#         self.qa_generator = QAPairGenerator(
#             policy_model=self.policy_model,
#             device=self.config.DEVICE
#         )
        
#         self.qa_storage = QAPairStorage(
#             output_dir=os.path.join(self.config.RESULTS_DIR, "qa_pairs")
#         )
        
#         logger.info("✓ Q-A Pair Generator initialized")
    
#     def _setup_evaluator(self):
#         """Setup evaluator"""
#         self.evaluator = Evaluator(self.config)
#         logger.info("✓ Evaluator initialized")
    
#     def train(self, num_iterations: int = None):
#         """Training loop - FIX #2: full_instruction properly defined"""
        
#         if num_iterations is None:
#             num_iterations = self.config.TRAINING_CONFIG['num_iterations']
        
#         logger.info("\n" + "=" * 80)
#         logger.info("STARTING TRAINING")
#         logger.info("=" * 80 + "\n")
        
#         # Error check
#         if self.api_executor is None:
#             logger.error("❌ api_executor not initialized!")
#             return
        
#         log_file = os.path.join(self.config.RESULTS_DIR, "training_metrics.json")
#         self.metrics_logger = MetricsLogger(log_file)
#         self.response_tracker = ResponseTracker()
        
#         train_tasks = self.data_loaders['train']
        
#         if not train_tasks:
#             logger.error("❌ NO TRAINING TASKS!")
#             return
        
#         logger.info(f"✓ Training with {len(train_tasks)} tasks\n")
        
#         for iteration in range(num_iterations):
#             logger.info(f"\n{'='*80}")
#             logger.info(f"ITERATION {iteration + 1}/{num_iterations}")
#             logger.info(f"{'='*80}\n")
            
#             iteration_metrics = {
#                 'iteration': iteration + 1,
#                 'num_tasks': len(train_tasks),
#                 'total_api_cost': 0,
#                 'avg_reward': 0,
#                 'completed_tasks': 0,
#             }
            
#             progress = ProgressTracker(len(train_tasks), f"Processing tasks (iter {iteration+1})")
            
#             for task_idx, task in enumerate(train_tasks):
#                 try:
#                     instruction = task.get('instruction', '')
#                     context = task.get('context', '')
#                     task_name = task.get('category', f'task_{task_idx+1}')
                    
#                     if not instruction.strip():
#                         progress.update(1)
#                         continue
                    
#                     # FIX #2: Define full_instruction BEFORE using it
#                     full_instruction = f"{instruction}\n{context}" if context else instruction
                    
#                     logger.info(f"\n  Task {task_idx + 1}/{len(train_tasks)}: {instruction[:50]}...")
                    
#                     # ============ BEFORE ============
#                     logger.info(f"  [BEFORE] Generating without context...")
#                     before_response = self.policy_model.generate(instruction)
#                     before_score = ImprovedScoringFunction.score_response(instruction, before_response)
                    
#                     logger.info(f"  ├─ Response: {before_response[:60]}...")
#                     logger.info(f"  └─ Score: {before_score:.4f}")
                    
#                     # ============ RETRIEVAL ============
#                     logger.info(f"\n  [RETRIEVAL] Fetching relevant data...")
                    
#                     api_results = self.api_executor.execute_trajectory(None, instruction)
#                     collected_data = api_results.get('collected_data', '')
                    
#                     logger.info(f"  ├─ Status: {api_results.get('debug', 'Unknown')}")
#                     if collected_data:
#                         logger.info(f"  └─ Data length: {len(collected_data)} chars")
#                     else:
#                         logger.warning(f"  └─ ⚠️ NO DATA RETRIEVED!")
                    
#                     # ============ Q-A GENERATION ============
#                     logger.info(f"\n  [Q-A GENERATION] Generating examples...")
                    
#                     if collected_data:
#                         qa_data = self.qa_generator.generate_qa_pairs(
#                             instruction=instruction,
#                             retrieved_data=collected_data,
#                             num_demonstrations=2,
#                             num_questions=1
#                         )
                        
#                         self.qa_storage.save_qa_pairs(
#                             task_number=task_idx + 1,
#                             task_name=task_name,
#                             qa_data=qa_data
#                         )
                        
#                         logger.info(f"  ├─ Demonstrations: {len(qa_data.get('demonstrations', []))}")
#                         logger.info(f"  └─ Q-A pairs: {len(qa_data.get('qa_pairs', []))}")
#                     else:
#                         logger.info(f"  └─ Skipped (no data)")
                    
#                     # ============ AFTER - WITH CONTEXT ============
#                     logger.info(f"\n  [AFTER] Generating WITH context...")
                    
#                     if collected_data:
#                         context_prompt = f"""Context Information:
# {collected_data[:500]}

# Question: {instruction}

# Answer:"""
                        
#                         after_response = self.policy_model.generate(
#                             instruction=context_prompt,
#                             max_tokens=256
#                         )
#                         after_score = ImprovedScoringFunction.score_response(instruction, after_response)
                        
#                         logger.info(f"  ├─ Response: {after_response[:60]}...")
#                         logger.info(f"  └─ Score: {after_score:.4f}")
#                     else:
#                         after_response = before_response
#                         after_score = before_score
#                         logger.info(f"  └─ (Using baseline)")
                    
#                     # ============ IMPROVEMENT ============
#                     improvement = after_score - before_score
#                     improvement_percent = (improvement / max(before_score, 0.01)) * 100
                    
#                     improvement_symbol = "⬆️" if improvement > 0.05 else "⬇️" if improvement < -0.05 else "→"
#                     logger.info(f"\n  {improvement_symbol} Improvement: {improvement:+.4f} ({improvement_percent:+.2f}%)")
                    
#                     # ============ TRACK ============
#                     self.response_tracker.add_comparison(
#                         task_number=task_idx + 1,
#                         instruction=instruction,
#                         before_response=before_response,
#                         before_score=before_score,
#                         after_response=after_response,
#                         after_score=after_score,
#                         retrieved_data=collected_data,
#                         full_instruction=full_instruction,  # FIX #2: Now defined!
#                         full_before_response=before_response,
#                         full_after_response=after_response
#                     )
                    
#                     iteration_metrics['total_api_cost'] += api_results.get('total_cost', 0)
#                     iteration_metrics['avg_reward'] += after_score
#                     iteration_metrics['completed_tasks'] += 1
                    
#                     progress.update(1)
                
#                 except Exception as e:
#                     logger.error(f"Task {task_idx + 1} failed: {e}", exc_info=True)
#                     progress.update(1)
            
#             progress.finish()
            
#             if iteration_metrics['completed_tasks'] > 0:
#                 iteration_metrics['avg_reward'] /= iteration_metrics['completed_tasks']
            
#             self.metrics_logger.log(**iteration_metrics)
            
#             logger.info(f"\n  Iteration Summary:")
#             logger.info(f"    - Tasks: {iteration_metrics['completed_tasks']}/{len(train_tasks)}")
#             logger.info(f"    - Avg reward: {iteration_metrics['avg_reward']:.4f}")
        
#         # Save results
#         logger.info("\n" + "=" * 80)
#         logger.info("SAVING RESULTS")
#         logger.info("=" * 80)
        
#         all_qa_file = self.qa_storage.save_all_qa_pairs()
#         logger.info(f"✓ Q-A pairs saved to {all_qa_file}")
        
#         self.metrics_logger.save()
#         self.response_tracker.save()
#         self.response_tracker.print_summary()
        
#         # FIX #3: Clear documents after training (optional cleanup)
#         self._cleanup_documents()
        
#         logger.info("\n" + "=" * 80)
#         logger.info("TRAINING COMPLETE")
#         logger.info("=" * 80)
    
#     def _cleanup_documents(self):
#         """Optional: Clear documents after training to free memory"""
#         try:
#             if self.retrieval_engine:
#                 self.retrieval_engine.hybrid_retriever.documents = []
#                 self.retrieval_engine.hybrid_retriever.document_embeddings = None
#             logger.info("✓ Documents cleared from memory")
#         except Exception as e:
#             logger.debug(f"Cleanup: {e}")
    
#     def evaluate(self):
#         """Evaluate on test set"""
#         logger.info("\n" + "=" * 80)
#         logger.info("EVALUATING ON TEST SET")
#         logger.info("=" * 80 + "\n")
        
#         test_tasks = self.data_loaders['test'][:3]
        
#         results = self.evaluator.evaluate_tasks(
#             policy_model=self.policy_model,
#             tasks=test_tasks,
#             api_executor=self.api_executor
#         )
        
#         results_file = os.path.join(self.config.RESULTS_DIR, "evaluation_results.json")
#         save_json(results, results_file)
        
#         logger.info("\n" + "=" * 80)
#         logger.info("EVALUATION COMPLETE")
#         logger.info("=" * 80)
        
#         return results
    
#     def save_checkpoint(self, path: str = None):
#         """Save checkpoint"""
#         if path is None:
#             path = os.path.join(self.config.RESULTS_DIR, "checkpoint.pt")
        
#         try:
#             checkpoint = {
#                 'model_name': self.policy_model.model_name,
#                 'model_state': self.policy_model.model.state_dict() if hasattr(self.policy_model.model, 'state_dict') else None,
#             }
            
#             torch.save(checkpoint, path)
#             logger.info(f"Checkpoint saved to {path}")
#         except Exception as e:
#             logger.warning(f"Failed to save checkpoint: {e}")
    
#     def run_full_pipeline(self):
#         """Run complete pipeline"""
#         logger.info("\n" + "=" * 80)
#         logger.info("RUNNING FULL ADS PIPELINE - FULLY CORRECTED")
#         logger.info("=" * 80 + "\n")
        
#         self.setup()
#         self.train(num_iterations=self.config.TRAINING_CONFIG['num_iterations'])
#         results = self.evaluate()
#         self.save_checkpoint()
        
#         logger.info("\n" + "=" * 80)
#         logger.info("FULL PIPELINE COMPLETE!")
#         logger.info("=" * 80)
        
#         return results


# def main():
#     """Main entry point"""
#     ads = ImprovedADSFramework(ADSConfig)
#     results = ads.run_full_pipeline()
#     return results


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.StreamHandler(sys.stdout),
#             logging.FileHandler('ads_framework.log')
#         ]
#     )
    
#     logger = logging.getLogger(__name__)
    
#     try:
#         results = main()
#         logger.info("✓ SUCCESS: ADS Framework execution completed")
#     except Exception as e:
#         logger.error(f"✗ FATAL ERROR: {e}", exc_info=True)
#         sys.exit(1)
#  main_FINAL_CORRECTED_COMPLETE.py
# COMPLETE WORKING VERSION - All fixes applied, ready to run

import logging
import os
import sys
import json
import torch
from typing import List, Dict, Any
from pathlib import Path
import wikipedia

from config import ADSConfig
from utils import DataCache, MetricsLogger, ProgressTracker, save_json, load_json
from data_loader import DataManager
from policy_model import ModelManager
from retrieval_engine import AdvancedRetrievalEngine
from evaluator import Evaluator
from response_tracker import ResponseTracker
from qa_pair_generator import QAPairGenerator, QAPairStorage

logger = logging.getLogger(__name__)


# ============================================================================
# WIKIPEDIA DOCUMENT LOADER (Real-time retrieval, no persistent storage)
# ============================================================================

class WikipediaDocumentLoader:
    """Load Wikipedia documents for real-time retrieval"""
    
    def __init__(self, cache_dir: str = "cache/wikipedia"):
        """Initialize document loader"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "wikipedia_docs.json"
        logger.info(f"[WikipediaLoader] Initialized")
    
    def load_documents(self, topics: List[str] = None, use_cache: bool = False) -> List[str]:
        """
        Load Wikipedia documents
        
        Args:
            topics: Specific topics to load
            use_cache: Whether to use cached documents (default: False for real-time)
        
        Returns:
            List of Wikipedia documents
        """
        if topics is None:
            topics = self._get_default_topics()
        
        # Try cache first if enabled
        if use_cache and self.cache_file.exists():
            logger.info(f"[WikipediaLoader] Loading from cache...")
            cached_docs = self._load_from_cache()
            if cached_docs:
                return cached_docs
        
        # Fetch fresh documents
        logger.info(f"[WikipediaLoader] Fetching {len(topics)} topics from Wikipedia...")
        documents = self._fetch_from_wikipedia(topics)
        
        # Don't save to cache by default (real-time retrieval)
        if use_cache and documents:
            self._save_to_cache(documents)
        
        return documents
    
    def _get_default_topics(self) -> List[str]:
        """Get default Wikipedia topics"""
        return [
            "Prime Minister of the United Kingdom",
            "Keir Starmer",
            "Academy Award for Best Picture",
            "Formula One World Championship",
            "iPhone",
            "Apple Inc.",
            "United Kingdom",
            "2025 Formula One season",
        ]
    
    def _fetch_from_wikipedia(self, topics: List[str]) -> List[str]:
        """Fetch documents from Wikipedia"""
        documents = []
        
        for topic in topics:
            try:
                results = wikipedia.search(topic, results=2)
                
                if results:
                    try:
                        page = wikipedia.page(results[0])
                        doc_text = page.content
                        
                        if len(doc_text) > 100:
                            documents.append(doc_text)
                            logger.debug(f"[WikipediaLoader] ✓ {results[0]}")
                    
                    except (wikipedia.exceptions.DisambiguationError, 
                           wikipedia.exceptions.PageError):
                        pass
            
            except Exception as e:
                logger.debug(f"[WikipediaLoader] Error fetching {topic}")
        
        logger.info(f"[WikipediaLoader] ✓ Fetched {len(documents)} documents")
        return documents
    
    def _save_to_cache(self, documents: List[str]):
        """Save documents to cache"""
        try:
            cache_data = {
                'timestamp': str(__import__('datetime').datetime.now()),
                'num_documents': len(documents),
                'documents': documents
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"[WikipediaLoader] Cached {len(documents)} docs")
        
        except Exception as e:
            logger.warning(f"[WikipediaLoader] Error saving cache: {e}")
    
    def _load_from_cache(self) -> List[str]:
        """Load documents from cache"""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            documents = cache_data.get('documents', [])
            logger.debug(f"[WikipediaLoader] Loaded {len(documents)} from cache")
            
            return documents
        
        except Exception as e:
            logger.debug(f"[WikipediaLoader] Error loading cache")
            return []


# ============================================================================
# API EXECUTOR (With proper error handling)
# ============================================================================

class ImprovedAPIExecutor:
    """Execute API with Advanced Retrieval Engine"""
    
    def __init__(self, retrieval_engine: AdvancedRetrievalEngine):
        """Initialize executor"""
        if retrieval_engine is None:
            raise ValueError("retrieval_engine cannot be None")
        
        self.retrieval_engine = retrieval_engine
        logger.info("[APIExecutor] ✓ Initialized")
    
    def execute_trajectory(self, trajectory, instruction: str = ""):
        """Execute trajectory using advanced retrieval"""
        
        if not instruction or not instruction.strip():
            return {
                'collected_data': '', 
                'total_cost': 0, 
                'api_calls': [], 
                'debug': 'Empty instruction'
            }
        
        try:
            results = self.retrieval_engine.retrieve(instruction, top_k=3)
            
            if results:
                collected_data = "\n\n".join(results)
            else:
                collected_data = ""
            
            return {
                'collected_data': collected_data,
                'total_cost': 1,
                'api_calls': ['advanced_retrieval'],
                'debug': f'Retrieved {len(results)} docs' if results else 'No results'
            }
        
        except Exception as e:
            logger.error(f"[APIExecutor] Error: {e}")
            return {
                'collected_data': '', 
                'total_cost': 0, 
                'api_calls': [], 
                'debug': f'Error: {str(e)}'
            }


# ============================================================================
# SCORING FUNCTION (Enhanced)
# ============================================================================

class ImprovedScoringFunction:
    """Improved answer scoring with multiple factors"""
    
    @staticmethod
    def score_response(instruction: str, response: str, context: str = "") -> float:
        """
        Score response using multiple factors
        
        Factors:
        - Non-empty response
        - Length quality (1 word vs 2-6 words vs longer)
        - Capitalization (proper nouns)
        - Context relevance
        """
        
        if not response or len(response.strip()) == 0:
            return 0.0
        
        response_lower = response.lower().strip()
        
        # Check for failure patterns
        fail_patterns = ['don\'t know', 'unclear', 'sorry', 'unable', 'unanswerable', 'unknown']
        if any(p in response_lower for p in fail_patterns):
            return 0.2
        
        # Length analysis
        words = response.split()
        word_count = len(words)
        has_capitals = any(c.isupper() for c in response)
        
        # Scoring based on length
        if word_count == 1:
            base_score = 0.7 if has_capitals else 0.4
        elif 2 <= word_count <= 6:
            base_score = 0.8 if has_capitals else 0.5
        elif 7 <= word_count <= 15:
            base_score = 0.75 if has_capitals else 0.55
        elif 16 <= word_count <= 30:
            base_score = 0.7
        else:
            base_score = 0.5
        
        # Context relevance bonus
        context_bonus = 0.0
        if context and len(context) > 50:
            # Check if response appears in context
            if response[:20].lower() in context.lower():
                context_bonus = 0.1
        
        # Final score
        final_score = min(0.95, base_score + context_bonus)
        
        return final_score


# ============================================================================
# MAIN ADS FRAMEWORK (Fully Corrected)
# ============================================================================

class ImprovedADSFramework:
    """
    ADS Framework - FULLY CORRECTED VERSION
    
    Key Fixes:
    1. api_executor properly initialized
    2. full_instruction properly defined
    3. Real-time document loading (no persistent cache)
    4. Proper error checking
    5. Enhanced scoring function
    6. All components working together
    """
    
    def __init__(self, config: ADSConfig = None):
        """Initialize framework"""
        if config is None:
            config = ADSConfig
        
        self.config = config
        self.data_manager = None
        self.model_manager = None
        self.retrieval_engine = None
        self.api_executor = None  # FIX #1: Initialize as None
        self.evaluator = None
        self.metrics_logger = None
        self.response_tracker = None
        self.qa_generator = None
        self.qa_storage = None
        
        logger.info("=" * 80)
        logger.info("INITIALIZING ADS FRAMEWORK - FULLY CORRECTED")
        logger.info("=" * 80)
        config.print_config()
    
    def setup(self):
        """Setup all components"""
        try:
            logger.info("\n[STEP 1/6] Loading Data...")
            self._setup_data()
            
            logger.info("\n[STEP 2/6] Initializing Models...")
            self._setup_models()
            
            logger.info("\n[STEP 3/6] Initializing Retrieval Engine...")
            self._setup_retrieval_engine()  # FIX #1: Includes APIExecutor
            
            logger.info("\n[STEP 4/6] LOADING WIKIPEDIA DOCUMENTS...")
            self._load_and_index_documents()
            
            logger.info("\n[STEP 5/6] Initializing Q-A Pair Generator...")
            self._setup_qa_generator()
            
            logger.info("\n[STEP 6/6] Initializing Evaluator...")
            self._setup_evaluator()
            
            logger.info("\n" + "=" * 80)
            logger.info("SETUP COMPLETE")
            logger.info("=" * 80 + "\n")
        
        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            raise
    
    def _setup_data(self):
        """Setup data manager and load datasets"""
        config_dict = {
            'dataset_config': self.config.DATASET_CONFIG,
            'wiki_docs': 0,
        }
        
        self.data_manager = DataManager(config_dict)
        self.data_manager.prepare_all_data()
        self.data_loaders = self.data_manager.get_data_loaders()
        self.data_manager.print_task_summary()
        
        train_tasks = self.data_loaders.get('train', [])
        logger.info(f"✓ Data loaded: {len(train_tasks)} train")
    
    def _setup_models(self):
        """Setup policy and optimizer models"""
        config_dict = {
            'policy_model_name': self.config.POLICY_MODEL_NAME,
            'device': self.config.DEVICE,
            'load_in_8bit': self.config.LOAD_IN_8BIT,
        }
        self.model_manager = ModelManager(config_dict)
        self.model_manager.initialize_models()
        
        self.policy_model = self.model_manager.get_policy_model()
        logger.info("✓ Models initialized")
    
    def _setup_retrieval_engine(self):
        """Setup retrieval engine and API executor"""
        # FIX #1: Initialize retrieval engine
        self.retrieval_engine = AdvancedRetrievalEngine()
        
        # FIX #1: INITIALIZE APIExecutor (was missing before!)
        self.api_executor = ImprovedAPIExecutor(self.retrieval_engine)
        
        logger.info("✓ Retrieval Engine + APIExecutor initialized")
    
    def _load_and_index_documents(self):
        """Load and index Wikipedia documents"""
        logger.info("\n" + "=" * 80)
        logger.info("LOADING WIKIPEDIA DOCUMENTS")
        logger.info("=" * 80)
        
        # Initialize loader
        doc_loader = WikipediaDocumentLoader()
        
        # Load documents (use_cache=False for real-time retrieval)
        wikipedia_documents = doc_loader.load_documents(use_cache=False)
        
        if not wikipedia_documents:
            logger.warning("⚠️ No Wikipedia documents loaded, using fallback...")
            wikipedia_documents = self._create_fallback_documents()
        
        # Index documents
        logger.info(f"[INDEXING] Indexing {len(wikipedia_documents)} documents...")
        self.retrieval_engine.index_documents(wikipedia_documents)
        logger.info(f"✓ Successfully indexed {len(wikipedia_documents)} documents!")
        
        logger.info("=" * 80)
    
    def _create_fallback_documents(self) -> List[str]:
        """Fallback documents if Wikipedia unavailable"""
        return [
            """Keir Starmer became Prime Minister of the United Kingdom on 5 July 2024.""",
            """Academy Award for Best Picture is the most prestigious award in the film industry.""",
            """Formula One World Championship is the highest class of international auto racing.""",
            """Apple Inc. produces the iPhone, a series of smartphones.""",
            """United Kingdom comprises England, Scotland, Wales and Northern Ireland.""",
        ]
    
    def _setup_qa_generator(self):
        """Setup Q-A pair generator and storage"""
        self.qa_generator = QAPairGenerator(
            policy_model=self.policy_model,
            device=self.config.DEVICE
        )
        
        self.qa_storage = QAPairStorage(
            output_dir=os.path.join(self.config.RESULTS_DIR, "qa_pairs")
        )
        
        logger.info("✓ Q-A Pair Generator initialized")
    
    def _setup_evaluator(self):
        """Setup evaluator"""
        self.evaluator = Evaluator(self.config)
        logger.info("✓ Evaluator initialized")
    
    def train(self, num_iterations: int = None):
        """
        Training loop with proper error handling
        
        FIX #2: full_instruction properly defined before use
        """
        
        if num_iterations is None:
            num_iterations = self.config.TRAINING_CONFIG['num_iterations']
        
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80 + "\n")
        
        # Error check
        if self.api_executor is None:
            logger.error("❌ api_executor not initialized!")
            return
        
        log_file = os.path.join(self.config.RESULTS_DIR, "training_metrics.json")
        self.metrics_logger = MetricsLogger(log_file)
        self.response_tracker = ResponseTracker()
        
        train_tasks = self.data_loaders['train']
        
        if not train_tasks:
            logger.error("❌ NO TRAINING TASKS!")
            return
        
        logger.info(f"✓ Training with {len(train_tasks)} tasks\n")
        
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
                    task_name = task.get('category', f'task_{task_idx+1}')
                    
                    if not instruction.strip():
                        progress.update(1)
                        continue
                    
                    # FIX #2: Define full_instruction BEFORE using it
                    full_instruction = f"{instruction}\n{context}" if context else instruction
                    
                    logger.info(f"\n  Task {task_idx + 1}/{len(train_tasks)}: {instruction[:50]}...")
                    
                    # ============ BEFORE (Baseline) ============
                    logger.info(f"  [BEFORE] Generating without context...")
                    before_response = self.policy_model.generate(instruction)
                    before_score = ImprovedScoringFunction.score_response(
                        instruction, 
                        before_response
                    )
                    
                    logger.info(f"  ├─ Response: {before_response[:60]}...")
                    logger.info(f"  └─ Score: {before_score:.4f}")
                    
                    # ============ RETRIEVAL ============
                    logger.info(f"\n  [RETRIEVAL] Fetching relevant data...")
                    
                    api_results = self.api_executor.execute_trajectory(None, instruction)
                    collected_data = api_results.get('collected_data', '')
                    
                    logger.info(f"  ├─ Status: {api_results.get('debug', 'Unknown')}")
                    if collected_data:
                        logger.info(f"  └─ Data length: {len(collected_data)} chars")
                    else:
                        logger.warning(f"  └─ ⚠️ NO DATA RETRIEVED!")
                    
                    # ============ Q-A GENERATION ============
                    logger.info(f"\n  [Q-A GENERATION] Generating examples...")
                    
                    if collected_data:
                        qa_data = self.qa_generator.generate_qa_pairs(
                            instruction=instruction,
                            retrieved_data=collected_data,
                            num_demonstrations=2,
                            num_questions=1
                        )
                        
                        self.qa_storage.save_qa_pairs(
                            task_number=task_idx + 1,
                            task_name=task_name,
                            qa_data=qa_data
                        )
                        
                        logger.info(f"  ├─ Demonstrations: {len(qa_data.get('demonstrations', []))}")
                        logger.info(f"  └─ Q-A pairs: {len(qa_data.get('qa_pairs', []))}")
                    else:
                        logger.info(f"  └─ Skipped (no data)")
                    
                    # ============ AFTER (With Context) ============
                    logger.info(f"\n  [AFTER] Generating WITH context...")
                    
                    if collected_data:
                        context_prompt = f"""Context Information:
{collected_data[:500]}

Question: {instruction}

Answer:"""
                        
                        after_response = self.policy_model.generate(
                            instruction=context_prompt,
                            max_tokens=256
                        )
                        after_score = ImprovedScoringFunction.score_response(
                            instruction, 
                            after_response,
                            collected_data
                        )
                        
                        logger.info(f"  ├─ Response: {after_response[:60]}...")
                        logger.info(f"  └─ Score: {after_score:.4f}")
                    else:
                        after_response = before_response
                        after_score = before_score
                        logger.info(f"  └─ (Using baseline)")
                    
                    # ============ IMPROVEMENT ============
                    improvement = after_score - before_score
                    improvement_percent = (improvement / max(before_score, 0.01)) * 100
                    
                    improvement_symbol = "⬆️" if improvement > 0.05 else "⬇️" if improvement < -0.05 else "→"
                    logger.info(f"\n  {improvement_symbol} Improvement: {improvement:+.4f} ({improvement_percent:+.2f}%)")
                    
                    # ============ TRACK (FIX #2: full_instruction now defined) ============
                    # self.response_tracker.add_comparison(
                    #     task_number=task_idx + 1,
                    #     instruction=instruction,
                    #     before_response=before_response,
                    #     before_score=before_score,
                    #     after_response=after_response,
                    #     after_score=after_score,
                    #     retrieved_data=collected_data,
                    #     full_instruction=full_instruction,  # FIX #2: Now defined!
                    #     full_before_response=before_response,
                    #     full_after_response=after_response
                    # )
                    self.response_tracker.add_comparison(
                        task_number=task_idx + 1,
                        instruction=instruction,
                        full_instruction=instruction,
                        before_response=before_response,
                        before_full_response=before_response,
                        before_score=before_score,
                        retrieved_data=collected_data,
                        after_response=after_response,
                        after_full_response=after_response,
                        after_score=after_score,
                    )

                    iteration_metrics['total_api_cost'] += api_results.get('total_cost', 0)
                    iteration_metrics['avg_reward'] += after_score
                    iteration_metrics['completed_tasks'] += 1
                    
                    progress.update(1)
                
                except Exception as e:
                    logger.error(f"Task {task_idx + 1} failed: {e}", exc_info=True)
                    progress.update(1)
            
            progress.finish()
            
            if iteration_metrics['completed_tasks'] > 0:
                iteration_metrics['avg_reward'] /= iteration_metrics['completed_tasks']
            
            self.metrics_logger.log(**iteration_metrics)
            
            logger.info(f"\n  Iteration Summary:")
            logger.info(f"    - Tasks: {iteration_metrics['completed_tasks']}/{len(train_tasks)}")
            logger.info(f"    - Avg reward: {iteration_metrics['avg_reward']:.4f}")
        
        # Save results
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        all_qa_file = self.qa_storage.save_all_qa_pairs()
        logger.info(f"✓ Q-A pairs saved to {all_qa_file}")
        
        self.metrics_logger.save()
        self.response_tracker.save()
        self.response_tracker.print_summary()
        
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
        """Save model checkpoint"""
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
        """Run complete ADS pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING FULL ADS PIPELINE - FULLY CORRECTED")
        logger.info("=" * 80 + "\n")
        
        self.setup()
        self.train(num_iterations=self.config.TRAINING_CONFIG['num_iterations'])
        results = self.evaluate()
        self.save_checkpoint()
        
        logger.info("\n" + "=" * 80)
        logger.info("FULL PIPELINE COMPLETE!")
        logger.info("=" * 80)
        
        return results


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    ads = ImprovedADSFramework(ADSConfig)
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
        logger.info("✓ SUCCESS: ADS Framework execution completed")
    except Exception as e:
        logger.error(f"✗ FATAL ERROR: {e}", exc_info=True)
        sys.exit(1)

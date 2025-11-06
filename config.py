# config_UPDATED.py
# UPDATED CONFIG - Supports custom tasks option

import os
from pathlib import Path


class ADSConfig:
    """
    ADS Framework Configuration
    
    NEW: Support for custom tasks
    - Set use_custom_tasks = True to use your own JSON file
    - Set custom_tasks_file = path to your tasks JSON
    """
    
    # ==================== PATHS ====================
    PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    CACHE_DIR = PROJECT_ROOT / "cache"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Create directories
    for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # ==================== DEVICE ====================
    DEVICE = "cpu"  # or "cuda" if GPU available
    NUM_WORKERS = 1
    MIXED_PRECISION = False
    
    # ==================== MODELS ====================
    POLICY_MODEL_NAME = "google/flan-t5-base"
    OPTIMIZER_MODEL_NAME = "google/flan-t5-base"
    
    # Generation parameters
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    DO_SAMPLE = True
    REPETITION_PENALTY = 1.1
    
    # Quantization
    LOAD_IN_8BIT = False
    LOAD_IN_4BIT = False
    
    # ==================== DATASET CONFIG ====================
    DATASET_CONFIG = {
        # NEW: Custom tasks support
        'use_custom_tasks': False,  # Set to True to use custom JSON
        'custom_tasks_file': 'data/custom_tasks.json',  # Path to your tasks
        
        # Dolly tasks (if use_custom_tasks = False)
        'total_tasks': 11,
        'train_tasks': 8,
        'test_tasks': 2,
        
        # Wikipedia (always on-demand)
        'wiki_docs': 0,
        
        # Task settings
        'instructions_per_task': 5,
        'observed_per_task': 3,
        'held_out_per_task': 2,
    }
    
    # ==================== WIKIPEDIA ====================
    WIKIPEDIA_CONFIG = {
        'use_cached': True,
        'cache_size': 2201,
        'chunk_size': 512,
        'stride': 256,
    }
    
    # ==================== API COSTS ====================
    API_CONFIG = {
        'information_retrieval': {
            'enabled': True,
            'use_bm25': True,
            'use_dense_retrieval': False,
            'top_k': 5,
            'cost': 1
        },
        'demonstration_generation': {
            'enabled': True,
            'num_samples': 3,
            'cost': 2
        },
        'question_answering': {
            'enabled': True,
            'use_local_model': True,
            'cost': 3
        }
    }
    
    API_COSTS = {
        'information_retrieval': 1,
        'demonstration_generation': 2,
        'question_answering': 3,
    }
    
    # ==================== TRAINING ====================
    TRAINING_CONFIG = {
        'approach': 'in_context_learning',
        'num_iterations': 2,
        'num_trajectories_per_task': 2,
        'use_default_trajectories': True,
        'num_default_trajectories': 1,
        'batch_size': 1,
        'max_sequence_length': 512,
        'warmup_ratio': 0.1,
        'learning_rate': 1e-5,
        'num_epochs': 1,
    }
    
    # ==================== OPTIMIZATION ====================
    RS_CONFIG = {
        'enabled': True,
        'num_iterations': 1,
        'temperature': 1.0,
    }
    
    DPO_CONFIG = {
        'enabled': True,
        'num_iterations': 1,
        'beta': 0.1,
    }
    
    # ==================== EVALUATION ====================
    EVAL_CONFIG = {
        'metrics': ['win_rate', 'tie_rate', 'loss_rate'],
        'sample_trajectories': 1,
        'eval_batch_size': 1,
        'use_reward_model': False,
        'use_gpt4_judge': False,
        'use_heuristic_scoring': True,
    }
    
    # ==================== BENCHMARKS ====================
    BENCHMARKS = {
        'alpaca_eval': {
            'enabled': True,
            'max_samples': 20
        },
        'arena_hard': {
            'enabled': False,
            'max_samples': 10
        },
        'mt_bench': {
            'enabled': False,
            'max_samples': 10
        }
    }
    
    # ==================== INFERENCE ====================
    INFERENCE_CONFIG = {
        'num_beams': 1,
        'early_stopping': False,
        'length_penalty': 1.0,
        'no_repeat_ngram_size': 3,
        'encoder_no_repeat_ngram_size': 0,
        'pad_token_id': 0,
    }
    
    # ==================== LOGGING ====================
    LOGGING_CONFIG = {
        'log_level': 'INFO',
        'log_dir': str(RESULTS_DIR / 'logs'),
        'save_interval': 50,
        'eval_interval': 10,
    }
    
    # ==================== OTHER ====================
    SEED = 42
    
    COST_CONTROL = {
        'enabled': True,
        'cost_tier': 0.1,
        'max_api_cost_per_trajectory': 10,
    }
    
    @classmethod
    def get_config_dict(cls):
        """Return config as dictionary"""
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("\n" + "=" * 80)
        print("ADS FRAMEWORK CONFIGURATION (WITH CUSTOM TASKS SUPPORT)")
        print("=" * 80)
        
        config_dict = cls.get_config_dict()
        for key, value in config_dict.items():
            if not callable(value):
                print(f"{key:40}: {value}")
        
        print("=" * 80)

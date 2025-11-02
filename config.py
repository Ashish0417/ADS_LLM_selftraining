# config_FINAL_FIXED.py - Corrected Configuration
# For CPU-Only Linux Laptop - NO GPU Required
# Project: "Let Large Language Models Find the Data to Train Themselves"

import os
from typing import Dict, List

class ADSConfig:
    """Master configuration for the ADS framework - CPU OPTIMIZED"""
    
    # ==================== ENVIRONMENT SETTINGS ====================
    DEVICE = "cpu"  # Force CPU mode
    NUM_WORKERS = 1  # Single worker for CPU stability
    MIXED_PRECISION = False  # Disabled for CPU
    
    # ==================== PATH CONFIGURATION ====================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, CACHE_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # ==================== MODEL CONFIGURATION ====================
    # ⚠️ CRITICAL: Use FLAN-T5-Base ONLY on CPU laptops
    # Do NOT use microsoft/phi-2 unless you have 8GB+ free RAM
    
    POLICY_MODEL_NAME = "google/flan-t5-base"  # ✅ WORKING - 250M, 1GB RAM
    # Alternative models:
    # "google/flan-t5-small"  # 80M - Ultra light, fastest
    # "google/flan-t5-large"  # 780M - Slower but better quality
    # "microsoft/phi-2"       # ❌ DON'T USE - Needs 3-4GB RAM
    
    OPTIMIZER_MODEL_NAME = POLICY_MODEL_NAME  # Initialize from policy
    
    # Model inference settings
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    DO_SAMPLE = True
    REPETITION_PENALTY = 1.1
    
    # Quantization for inference
    LOAD_IN_8BIT = False  # ✅ Disabled for FLAN-T5 (doesn't need it)
    LOAD_IN_4BIT = False  # Disabled for CPU
    
    # ==================== DATASET CONFIGURATION ====================
    # Reduced dataset sizes for CPU laptop
    DATASET_CONFIG = {
        "total_tasks": 100,  # Total instructions (was 10,239 in paper)
        "train_tasks": 80,   # Training tasks
        "valid_tasks": 10,   # Validation tasks
        "test_tasks": 10,    # Test tasks
        "instructions_per_task": 5,  # Per task variants
        "observed_per_task": 3,      # Observed examples
        "held_out_per_task": 2,      # Held-out examples
    }
    
    # Wikipedia subset configuration
    WIKIPEDIA_CONFIG = {
        "use_cached": True,
        "cache_size": 2201,  # From real Wikipedia load (2201 docs loaded successfully)
        "chunk_size": 512,
        "stride": 256,
    }
    
    # ==================== API CONFIGURATION ====================
    API_CONFIG = {
        "information_retrieval": {
            "enabled": True,
            "use_bm25": True,
            "use_dense_retrieval": False,  # Skip dense retrieval on CPU
            "top_k": 5,
            "cost": 1,
        },
        "demonstration_generation": {
            "enabled": True,
            "num_samples": 3,
            "cost": 2,
        },
        "question_answering": {
            "enabled": True,
            "use_local_model": True,  # Use local LLM instead of external API
            "cost": 3,
        }
    }
    
    API_COSTS = {
        "information_retrieval": 1,
        "demonstration_generation": 2,
        "question_answering": 3,
    }
    
    # ==================== TRAINING CONFIGURATION ====================
    TRAINING_CONFIG = {
        "approach": "in_context_learning",  # Not fine-tuning to save memory
        "num_iterations": 2,  # Reduced iterations for CPU
        "num_trajectories_per_task": 2,  # Reduced from 5
        "use_default_trajectories": True,
        "num_default_trajectories": 1,
        "batch_size": 1,  # CPU limitation - MUST be 1
        "max_sequence_length": 512,  # Reduced from 1024 for CPU memory
        "warmup_ratio": 0.1,
        "learning_rate": 1e-5,
        "num_epochs": 1,
    }
    
    # RS and DPO settings
    RS_CONFIG = {
        "enabled": True,
        "num_iterations": 1,
        "temperature": 1.0,
    }
    
    DPO_CONFIG = {
        "enabled": True,
        "num_iterations": 1,
        "beta": 0.1,
    }
    
    # ==================== EVALUATION CONFIGURATION ====================
    EVAL_CONFIG = {
        "metrics": ["win_rate", "tie_rate", "loss_rate"],
        "sample_trajectories": 1,
        "eval_batch_size": 1,
        "use_reward_model": False,  # Skip reward model on CPU
        "use_gpt4_judge": False,    # Use fallback heuristic evaluation
        "use_heuristic_scoring": True,  # Use simple scoring function
    }
    
    # ==================== BENCHMARKS ====================
    BENCHMARKS = {
        "alpaca_eval": {
            "enabled": True,
            "max_samples": 20,  # Reduced from 805
        },
        "arena_hard": {
            "enabled": False,  # Skip resource-intensive benchmarks
            "max_samples": 10,
        },
        "mt_bench": {
            "enabled": False,
            "max_samples": 10,
        }
    }
    
    # ==================== INFERENCE SETTINGS ====================
    INFERENCE_CONFIG = {
        "num_beams": 1,  # Greedy decoding instead of beam search
        "early_stopping": False,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "encoder_no_repeat_ngram_size": 0,
        "pad_token_id": 0,
    }
    
    # ==================== LOGGING & CHECKPOINTING ====================
    LOGGING_CONFIG = {
        "log_level": "INFO",
        "log_dir": os.path.join(RESULTS_DIR, "logs"),
        "save_interval": 50,
        "eval_interval": 10,
    }
    
    # ==================== REPRODUCIBILITY ====================
    SEED = 42
    
    # ==================== COST CONTROL ====================
    COST_CONTROL = {
        "enabled": True,
        "cost_tier": 0.1,
        "max_api_cost_per_trajectory": 10,
    }
    
    @classmethod
    def get_config_dict(cls) -> Dict:
        """Get configuration as dictionary for logging"""
        config_dict = {}
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not callable(value):
                config_dict[key] = value
        return config_dict
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("\n" + "="*80)
        print("ADS FRAMEWORK CONFIGURATION (CPU-OPTIMIZED)")
        print("="*80)
        for key, value in cls.get_config_dict().items():
            if not callable(value):
                print(f"{key:40s}: {value}")
        print("="*80 + "\n")


# ============================================================================
# CRITICAL SETTINGS FOR YOUR LAPTOP
# ============================================================================

# Your log shows:
# 1. ✅ Wikitext successfully loaded 2201 documents (GOOD!)
# 2. ❌ Phi-2 model killed during loading (OOM - Out of Memory)
# 3. ✅ FLAN-T5-Base should work (250M, only needs 1GB RAM)

# REMEMBER:
# - FLAN-T5-Base: 250M params, 1GB RAM ✅ WORKS
# - Phi-2: 2.7B params, 3-4GB RAM ❌ GETS KILLED
# - Never load Phi-2 unless you have 8GB+ free RAM

# To use Phi-2, you would need:
# - Reduce max_sequence_length to 256
# - Reduce total_tasks to 10
# - Reduce cache_size to 500
# - Use quantization (but still risky)

# For now: STICK WITH FLAN-T5-BASE!

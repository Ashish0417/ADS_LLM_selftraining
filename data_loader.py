# data_loader_FLEXIBLE.py
# FLEXIBLE DATA LOADER - Supports both Dolly and custom JSON tasks
# FIXED: Uses percentage-based train/test splits (not absolute counts)

import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset

logger = logging.getLogger(__name__)


class DataManager:
    """
    Flexible Data Manager - Load from Dolly OR custom JSON file
    
    FIXED: Percentage-based train/test splits
    - 80% training, 20% testing by default
    - Configurable via config dictionary
    - Works for any number of tasks (1 to 10000+)
    
    Usage:
    1. Dolly (default): Loads from HuggingFace
    2. Custom JSON: Loads from your own file
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize data manager"""
        self.config = config_dict
        self.cache_dir = Path("cache")
        self.data_dir = Path("data")
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.data_loaders = {}
        
        # Check if using custom tasks
        self.use_custom_tasks = config_dict.get('use_custom_tasks', True)
        self.custom_tasks_file = config_dict.get('custom_tasks_file', 'data/custom_tasks.json')
        
        # ============ NEW: Percentage-based split configuration ============
        self.train_percentage = config_dict.get('train_percentage', 0.8)  # 80% for training
        self.test_percentage = config_dict.get('test_percentage', 0.2)   # 20% for testing
        
        logger.info(f"[DataManager] Initialized")
        logger.info(f"[DataManager] Mode: {'CUSTOM JSON' if self.use_custom_tasks else 'DOLLY'}")
        logger.info(f"[DataManager] Train/Test Split: {self.train_percentage*100:.0f}% / {self.test_percentage*100:.0f}%")
    
    def prepare_all_data(self):
        """Prepare all data"""
        logger.info("[DataManager] Preparing data...")
        
        if self.use_custom_tasks:
            # Load from custom JSON file
            self._load_custom_tasks()
        else:
            # Load from Dolly (default)
            self._load_dolly_tasks()
        
        logger.info("[DataManager] ✓ Data preparation complete")
    
    def _load_custom_tasks(self):
        """
        Load tasks from custom JSON file with percentage-based splits
        
        Expected JSON format:
        {
          "tasks": [
            {
              "instruction": "Your question here",
              "context": "Optional context",
              "response": "Expected answer (optional)",
              "category": "Optional category"
            },
            ...
          ]
        }
        """
        logger.info(f"[DataManager] Loading custom tasks from: {self.custom_tasks_file}")
        
        try:
            custom_file = Path(self.custom_tasks_file)
            
            if not custom_file.exists():
                logger.error(f"[DataManager] Custom tasks file not found: {self.custom_tasks_file}")
                logger.info(f"[DataManager] Creating example file: {self.custom_tasks_file}")
                self._create_example_custom_tasks()
                logger.info(f"[DataManager] Please edit {self.custom_tasks_file} and re-run")
                self.data_loaders['train'] = []
                self.data_loaders['test'] = []
                return
            
            # Load JSON
            with open(custom_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            all_tasks = data.get('tasks', [])
            
            if not all_tasks:
                logger.warning("[DataManager] No tasks found in custom file!")
                self.data_loaders['train'] = []
                self.data_loaders['test'] = []
                return
            
            logger.info(f"[DataManager] Loaded {len(all_tasks)} custom tasks")
            
            # Validate task format
            validated_tasks = []
            for i, task in enumerate(all_tasks):
                if not isinstance(task, dict):
                    logger.warning(f"[DataManager] Task {i} is not a dict, skipping")
                    continue
                
                if 'instruction' not in task:
                    logger.warning(f"[DataManager] Task {i} missing 'instruction', skipping")
                    continue
                
                # Ensure all fields exist
                validated_task = {
                    'instruction': task['instruction'],
                    'context': task.get('context', ''),
                    'response': task.get('response', ''),
                    'category': task.get('category', 'custom')
                }
                validated_tasks.append(validated_task)
            
            logger.info(f"[DataManager] Validated {len(validated_tasks)}/{len(all_tasks)} tasks")
            
            # ============ FIXED: Use percentage-based split ============
            total_tasks = len(validated_tasks)
            
            # Calculate split point using percentage
            split_point = int(total_tasks * self.train_percentage)
            
            # Ensure at least 1 task for training if possible
            # If only 1 task, use for training only
            if total_tasks == 1:
                split_point = 1
                logger.warning("[DataManager] ⚠️ Only 1 task provided, using for training")
            elif total_tasks == 2:
                split_point = 1  # 50/50 split
            else:
                # Ensure split point is valid
                split_point = max(1, min(split_point, total_tasks - 1))
            
            # Split using calculated split point
            train_tasks = validated_tasks[:split_point]
            test_tasks = validated_tasks[split_point:]
            
            self.data_loaders['train'] = train_tasks
            self.data_loaders['test'] = test_tasks
            
            # Log with percentages
            train_pct = (len(train_tasks) / total_tasks * 100) if total_tasks > 0 else 0
            test_pct = (len(test_tasks) / total_tasks * 100) if total_tasks > 0 else 0
            
            logger.info(f"[DataManager] ✓ Split: {len(train_tasks)} train ({train_pct:.1f}%), {len(test_tasks)} test ({test_pct:.1f}%)")
            
        except json.JSONDecodeError as e:
            logger.error(f"[DataManager] JSON parsing error: {e}")
            logger.error(f"[DataManager] Please check JSON syntax in {self.custom_tasks_file}")
            self.data_loaders['train'] = []
            self.data_loaders['test'] = []
        except Exception as e:
            logger.error(f"[DataManager] Error loading custom tasks: {e}")
            self.data_loaders['train'] = []
            self.data_loaders['test'] = []
    
    def _create_example_custom_tasks(self):
        """Create an example custom tasks file"""
        example_tasks = {
            "tasks": [
                {
                    "instruction": "When did Virgin Australia start operating?",
                    "context": "Virgin Australia, the trading name of Virgin Australia Airlines Pty Ltd, is an Australian-based airline. It commenced services on 31 August 2000 as Virgin Blue.",
                    "response": "31 August 2000",
                    "category": "factual"
                },
                {
                    "instruction": "Which is a species of fish? Tope or Rope",
                    "context": "",
                    "response": "Tope",
                    "category": "trivia"
                },
                {
                    "instruction": "Why can camels survive for long without water?",
                    "context": "",
                    "response": "Camels have humps that store fat which can be converted to water and energy when needed",
                    "category": "science"
                },
                {
                    "instruction": "Alice's parents have three daughters: Amy, Jessy, and what's the name of the third daughter?",
                    "context": "",
                    "response": "Alice",
                    "category": "riddle"
                },
                {
                    "instruction": "What is the capital of France?",
                    "context": "",
                    "response": "Paris",
                    "category": "geography"
                }
            ]
        }
        
        try:
            output_file = Path(self.custom_tasks_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(example_tasks, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[DataManager] ✓ Created example file: {self.custom_tasks_file}")
        except Exception as e:
            logger.error(f"[DataManager] Failed to create example file: {e}")
    
    def _load_dolly_tasks(self):
        """Load Dolly instruction dataset from HuggingFace"""
        logger.info("[DataManager] Loading Dolly instruction tasks...")
        
        try:
            # Check cache first
            if self._load_from_cache():
                logger.info("[DataManager] ✓ Loaded from cache")
                return
            
            # Load from HuggingFace
            dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
            
            # Convert to our task format
            all_tasks = []
            for item in dataset:
                task = {
                    'instruction': item['instruction'],
                    'context': item.get('context', ''),
                    'response': item['response'],
                    'category': item.get('category', 'general')
                }
                all_tasks.append(task)
            
            logger.info(f"[DataManager] Loaded {len(all_tasks)} Dolly tasks")
            
            # ============ FIXED: Use percentage-based split for Dolly too ============
            total_tasks = self.config.get('total_tasks', 100)
            all_tasks = all_tasks[:total_tasks]
            
            # Calculate split using percentage
            split_point = int(len(all_tasks) * self.train_percentage)
            split_point = max(1, min(split_point, len(all_tasks) - 1))
            
            # Split
            train_tasks = all_tasks[:split_point]
            test_tasks = all_tasks[split_point:]
            
            self.data_loaders['train'] = train_tasks
            self.data_loaders['test'] = test_tasks
            
            # Log with percentages
            train_pct = (len(train_tasks) / len(all_tasks) * 100) if len(all_tasks) > 0 else 0
            test_pct = (len(test_tasks) / len(all_tasks) * 100) if len(all_tasks) > 0 else 0
            
            logger.info(f"[DataManager] ✓ Split: {len(train_tasks)} train ({train_pct:.1f}%), {len(test_tasks)} test ({test_pct:.1f}%)")
            
            # Save to cache
            self._save_to_cache(train_tasks, test_tasks)
            
        except Exception as e:
            logger.error(f"[DataManager] Failed to load Dolly tasks: {e}")
            self.data_loaders['train'] = []
            self.data_loaders['test'] = []
    
    def _save_to_cache(self, train_tasks: List[Dict], test_tasks: List[Dict]):
        """Save tasks to cache"""
        try:
            train_file = self.data_dir / "tasks_train.json"
            test_file = self.data_dir / "tasks_test.json"
            
            with open(train_file, 'w', encoding='utf-8') as f:
                json.dump({'tasks': train_tasks}, f, ensure_ascii=False, indent=2)
            
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump({'tasks': test_tasks}, f, ensure_ascii=False, indent=2)
            
            logger.info("[DataManager] ✓ Saved to cache")
        except Exception as e:
            logger.warning(f"[DataManager] Failed to save cache: {e}")
    
    def _load_from_cache(self) -> bool:
        """Try to load tasks from cache"""
        try:
            train_file = self.data_dir / "tasks_train.json"
            test_file = self.data_dir / "tasks_test.json"
            
            if train_file.exists() and test_file.exists():
                with open(train_file, encoding='utf-8') as f:
                    train_data = json.load(f)
                    train_tasks = train_data.get('tasks', [])
                
                with open(test_file, encoding='utf-8') as f:
                    test_data = json.load(f)
                    test_tasks = test_data.get('tasks', [])
                
                if train_tasks and test_tasks:
                    self.data_loaders['train'] = train_tasks
                    self.data_loaders['test'] = test_tasks
                    
                    # Log with percentages
                    total = len(train_tasks) + len(test_tasks)
                    train_pct = (len(train_tasks) / total * 100) if total > 0 else 0
                    test_pct = (len(test_tasks) / total * 100) if total > 0 else 0
                    
                    logger.info(f"[DataManager] Loaded from cache: {len(train_tasks)} train ({train_pct:.1f}%), {len(test_tasks)} test ({test_pct:.1f}%)")
                    return True
        except Exception as e:
            logger.debug(f"[DataManager] Cache load failed: {e}")
        
        return False
    
    def load_wikipedia_documents(self) -> List[str]:
        """Wikipedia fetched on-demand"""
        logger.info("[DataManager] Wikipedia: ON-DEMAND")
        return []
    
    def get_data_loaders(self) -> Dict[str, List]:
        """Get all data loaders"""
        return self.data_loaders
    
    def print_task_summary(self):
        """Print summary of loaded tasks"""
        train_tasks = self.data_loaders.get('train', [])
        test_tasks = self.data_loaders.get('test', [])
        
        total = len(train_tasks) + len(test_tasks)
        train_pct = (len(train_tasks) / total * 100) if total > 0 else 0
        test_pct = (len(test_tasks) / total * 100) if total > 0 else 0
        
        logger.info("\n" + "=" * 80)
        logger.info("TASK SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Mode: {'CUSTOM JSON' if self.use_custom_tasks else 'DOLLY'}")
        if self.use_custom_tasks:
            logger.info(f"Custom file: {self.custom_tasks_file}")
        logger.info(f"Train tasks: {len(train_tasks)} ({train_pct:.1f}%)")
        logger.info(f"Test tasks: {len(test_tasks)} ({test_pct:.1f}%)")
        logger.info(f"Total tasks: {total}")
        
        if train_tasks:
            logger.info("\nFirst 3 train tasks:")
            for i, task in enumerate(train_tasks[:3], 1):
                logger.info(f"  {i}. {task['instruction'][:60]}...")
        
        logger.info("=" * 80)

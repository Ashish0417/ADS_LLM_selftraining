# data_loader_FIXED_DOLLY.py
# FIXED DATA LOADER - Loads Dolly instruction tasks correctly

import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset

logger = logging.getLogger(__name__)


class DataManager:
    """Updated Data Manager with Dolly task loading"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize data manager"""
        self.config = config_dict
        self.cache_dir = Path("cache")
        self.data_dir = Path("data")
        self.cache_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.data_loaders = {}
        logger.info("[DataManager] Initialized (Wikipedia: ON-DEMAND)")
    
    def prepare_all_data(self):
        """Prepare all data"""
        logger.info("[DataManager] Preparing data...")
        
        # Load Dolly instruction tasks
        self._load_dolly_tasks()
        
        logger.info("[DataManager] ✓ Data preparation complete")
    
    def _load_dolly_tasks(self):
        """Load Dolly instruction dataset from HuggingFace"""
        logger.info("[DataManager] Loading Dolly instruction tasks...")
        
        try:
            # Load Dolly 15k dataset from HuggingFace
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
            
            # Split into train/test
            total_tasks = self.config.get('total_tasks', 100)
            test_split = self.config.get('test_tasks', 10)
            
            # Limit to configured number
            all_tasks = all_tasks[:total_tasks]
            
            # Split
            train_tasks = all_tasks[:-test_split]
            test_tasks = all_tasks[-test_split:]
            
            self.data_loaders['train'] = train_tasks
            self.data_loaders['test'] = test_tasks
            
            logger.info(f"[DataManager] Split into {len(train_tasks)} train, {len(test_tasks)} test tasks")
            
            # Save to cache for faster loading next time
            self._save_tasks_to_cache(train_tasks, test_tasks)
        
        except Exception as e:
            logger.error(f"[DataManager] Failed to load Dolly tasks: {e}")
            # Fallback to empty
            self.data_loaders['train'] = []
            self.data_loaders['test'] = []
    
    def _save_tasks_to_cache(self, train_tasks: List[Dict], test_tasks: List[Dict]):
        """Save tasks to cache files"""
        try:
            train_file = self.data_dir / "tasks_train.json"
            test_file = self.data_dir / "tasks_test.json"
            
            with open(train_file, 'w') as f:
                json.dump({'tasks': train_tasks}, f)
            
            with open(test_file, 'w') as f:
                json.dump({'tasks': test_tasks}, f)
            
            logger.info("[DataManager] ✓ Saved tasks to cache")
        except Exception as e:
            logger.warning(f"[DataManager] Failed to save cache: {e}")
    
    def _load_tasks_from_cache(self) -> bool:
        """Try to load tasks from cache files"""
        try:
            train_file = self.data_dir / "tasks_train.json"
            test_file = self.data_dir / "tasks_test.json"
            
            if train_file.exists() and test_file.exists():
                with open(train_file) as f:
                    train_data = json.load(f)
                    train_tasks = train_data.get('tasks', [])
                
                with open(test_file) as f:
                    test_data = json.load(f)
                    test_tasks = test_data.get('tasks', [])
                
                if train_tasks and test_tasks:
                    self.data_loaders['train'] = train_tasks
                    self.data_loaders['test'] = test_tasks
                    logger.info(f"[DataManager] ✓ Loaded from cache: {len(train_tasks)} train, {len(test_tasks)} test")
                    return True
        except Exception as e:
            logger.debug(f"[DataManager] Cache load failed: {e}")
        
        return False
    
    def load_wikipedia_documents(self) -> List[str]:
        """
        Return empty list - Wikipedia fetched on-demand!
        """
        logger.info("[DataManager] Wikipedia: ON-DEMAND FETCHING")
        return []
    
    def get_data_loaders(self) -> Dict[str, List]:
        """Get all data loaders"""
        return self.data_loaders

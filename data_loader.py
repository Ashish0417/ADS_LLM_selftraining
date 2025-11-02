import logging
import json
import os
from typing import List, Dict, Tuple, Any
from datasets import load_dataset
import numpy as np
from utils import DataCache, ProgressTracker, TextProcessor

logger = logging.getLogger(__name__)


class WikipediaDataLoader:
    """Load Wikipedia data efficiently for CPU"""
    
    def __init__(self, cache_dir: str = "cache", max_docs: int = 5000):
        self.cache = DataCache(cache_dir)
        self.max_docs = max_docs
        self.documents = []
    
    def load_wikipedia_subset(self) -> List[str]:
        """Load Wikipedia subset"""
        cache_key = f"wikipedia_subset_{self.max_docs}"
        cached = self.cache.load(cache_key)
        if cached is not None and len(cached) > 0:  # FIX: Only use non-empty cache
            logger.info(f"Loaded {len(cached)} documents from cache")
            return cached
        
        try:
            logger.info(f"Loading Wikipedia subset ({self.max_docs} documents)...")
            
            # Load from HuggingFace - Wikitext works reliably
            dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
            
            documents = []
            progress = ProgressTracker(self.max_docs, "Loading Wikipedia")
            
            for i, example in enumerate(dataset):
                if i >= self.max_docs:
                    break
                
                text = example.get('text', '')
                if text and len(text) > 50:
                    documents.append(TextProcessor.clean_text(text))
                
                progress.update(1)
            
            progress.finish()
            self.cache.save(cache_key, documents)
            
            logger.info(f"Loaded {len(documents)} Wikipedia documents")
            return documents
        
        except Exception as e:
            logger.error(f"Failed to load Wikipedia: {e}")
            logger.info("Using dummy documents for demonstration")
            return self._get_dummy_documents()
    
    def _get_dummy_documents(self) -> List[str]:
        """Get dummy documents for testing"""
        dummy_docs = [
            "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
            "Machine Learning is a subset of AI that enables systems to learn from data.",
            "Natural Language Processing helps computers understand and process human language.",
            "Deep Learning uses neural networks with multiple layers for pattern recognition.",
            "Quantum Computing uses quantum mechanics principles for computation.",
            "Neural Networks are inspired by biological neural networks in animal brains.",
            "Transfer Learning applies knowledge from one task to another task.",
            "Computer Vision enables machines to interpret visual information from images.",
        ]
        
        multiplied = dummy_docs * (self.max_docs // len(dummy_docs) + 1)
        return multiplied[:self.max_docs]


class MagpieDataLoader:
    """Load instruction dataset - FIXED WITH CACHE BYPASS"""
    
    def __init__(self, cache_dir: str = "cache", max_tasks: int = 100):
        self.cache = DataCache(cache_dir)
        self.max_tasks = max_tasks
    
    def load_magpie_subset(self) -> List[Dict]:
        """Load instruction dataset from reliable sources"""
        cache_key = f"magpie_subset_{self.max_tasks}"
        cached = self.cache.load(cache_key)
        
        # FIX: Only use cache if it has data (non-empty)
        if cached is not None and len(cached) > 0:
            logger.info(f"Loaded {len(cached)} tasks from cache")
            return cached
        
        logger.info(f"No valid cache found (or cache is empty). Loading fresh data...")
        
        # Try multiple datasets in order of reliability
        datasets_to_try = [
            ("sahil2801/alpaca_gpt4_data", "Alpaca-GPT4"),
            ("databricks/databricks-dolly-15k", "Dolly"),
            ("GAIR/lima", "LIMA"),
        ]
        
        for dataset_name, friendly_name in datasets_to_try:
            try:
                logger.info(f"Attempting to load {friendly_name}...")
                tasks = self._load_from_dataset(dataset_name, friendly_name)
                
                if len(tasks) > 0:
                    self.cache.save(cache_key, tasks)
                    logger.info(f"Successfully loaded {len(tasks)} tasks from {friendly_name}")
                    return tasks
            
            except Exception as e:
                logger.warning(f"Failed to load {friendly_name}: {e}")
                continue
        
        # If all fail, use dummy
        logger.error("All datasets failed to load")
        logger.info("Using dummy instructions for demonstration")
        return self._get_dummy_instructions()
    
    def _load_from_dataset(self, dataset_name: str, friendly_name: str) -> List[Dict]:
        """Load from specific dataset"""
        logger.info(f"Loading {friendly_name} ({self.max_tasks} tasks)...")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        tasks = []
        progress = ProgressTracker(min(self.max_tasks, 200), f"Loading {friendly_name}")
        
        for i, example in enumerate(dataset):
            if len(tasks) >= self.max_tasks:
                break
            
            # Handle different dataset formats
            instruction = None
            response = None
            
            if 'instruction' in example:
                instruction = example['instruction']
                response = example.get('output', example.get('response', ''))
            
            elif 'question' in example:
                instruction = example['question']
                response = example.get('answer', '')
            
            elif 'conversations' in example:
                convs = example['conversations']
                if len(convs) >= 2:
                    instruction = convs[0] if isinstance(convs[0], str) else convs[0].get('value', '')
                    response = convs[1] if isinstance(convs[1], str) else convs[1].get('value', '')
            
            # Add if valid
            if instruction and response and len(instruction.strip()) > 0 and len(response.strip()) > 0:
                tasks.append({
                    'instruction': instruction.strip(),
                    'response': response.strip(),
                    'category': example.get('category', 'general'),
                    'difficulty': 'medium',
                })
            
            progress.update(1)
        
        progress.finish()
        return tasks
    
    def _get_dummy_instructions(self) -> List[Dict]:
        """Get dummy instructions for testing"""
        dummy_instructions = [
            {"instruction": "Explain machine learning", "response": "Machine Learning is learning from data", 
             "category": "education", "difficulty": "easy"},
            {"instruction": "What is quantum computing?", "response": "Quantum computing uses quantum mechanics", 
             "category": "education", "difficulty": "medium"},
            {"instruction": "How to solve a quadratic equation?", "response": "Use the quadratic formula", 
             "category": "math", "difficulty": "medium"},
            {"instruction": "Write a Python function to sort a list", "response": "def sort_list(lst): return sorted(lst)", 
             "category": "coding", "difficulty": "easy"},
            {"instruction": "Explain neural networks", "response": "Networks inspired by biological brains", 
             "category": "education", "difficulty": "hard"},
        ]
        
        multiplied = dummy_instructions * (self.max_tasks // len(dummy_instructions) + 1)
        return multiplied[:self.max_tasks]


class InstructionDataset:
    """Manage instruction dataset for training"""
    
    def __init__(self, instructions: List[Dict], config: Dict):
        self.instructions = instructions
        self.config = config
        self.task_clusters = {}
        self._create_task_clusters()
    
    def _create_task_clusters(self):
        """Cluster instructions by category and difficulty"""
        for instruction in self.instructions:
            category = instruction.get('category', 'general')
            difficulty = instruction.get('difficulty', 'medium')
            
            key = f"{category}_{difficulty}"
            
            if key not in self.task_clusters:
                self.task_clusters[key] = []
            
            self.task_clusters[key].append(instruction)
        
        logger.info(f"Created {len(self.task_clusters)} task clusters")
    
    def get_task_split(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split instructions into train/valid/test"""
        train_tasks = []
        valid_tasks = []
        test_tasks = []
        
        train_count = self.config.get('train_tasks', 80)
        valid_count = self.config.get('valid_tasks', 10)
        test_count = self.config.get('test_tasks', 10)
        
        # Randomly distribute
        np.random.shuffle(self.instructions)
        
        train_tasks = self.instructions[:train_count]
        valid_tasks = self.instructions[train_count:train_count+valid_count]
        test_tasks = self.instructions[train_count+valid_count:train_count+valid_count+test_count]
        
        logger.info(f"Dataset split: train={len(train_tasks)}, valid={len(valid_tasks)}, test={len(test_tasks)}")
        
        return train_tasks, valid_tasks, test_tasks
    
    def get_observed_and_held_out(self, tasks: List[Dict], obs_count: int = 3, 
                                   held_out_count: int = 2) -> List[Dict]:
        """Split each task into observed and held-out"""
        split_tasks = []
        
        for task in tasks:
            task_variants = []
            for i in range(obs_count + held_out_count):
                variant = task.copy()
                variant['split_id'] = i
                task_variants.append(variant)
            
            split_tasks.extend(task_variants)
        
        return split_tasks


class BenchmarkDataLoader:
    """Load evaluation benchmarks"""
    
    @staticmethod
    def load_alpaca_eval(max_samples: int = 20) -> List[Dict]:
        """Load AlpacaEval 2.0"""
        try:
            logger.info("Loading AlpacaEval 2.0...")
            dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")
            
            samples = []
            for i, example in enumerate(dataset):
                if i >= max_samples:
                    break
                samples.append({
                    'instruction': example.get('instruction', ''),
                    'input': example.get('input', ''),
                    'category': example.get('category', 'general'),
                })
            
            logger.info(f"Loaded {len(samples)} AlpacaEval samples")
            return samples
        except Exception as e:
            logger.warning(f"Failed to load AlpacaEval: {e}")
            return []


class DataManager:
    """Unified data management"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.wikipedia_docs = []
        self.magpie_tasks = []
        self.instruction_dataset = None
    
    def prepare_all_data(self):
        """Prepare all required data"""
        logger.info("=" * 80)
        logger.info("PREPARING DATA FOR ADS FRAMEWORK")
        logger.info("=" * 80)
        
        # Load Wikipedia
        wiki_loader = WikipediaDataLoader(max_docs=self.config.get('wiki_docs', 5000))
        self.wikipedia_docs = wiki_loader.load_wikipedia_subset()
        
        # Load Instruction dataset (with fallbacks and cache bypass)
        magpie_loader = MagpieDataLoader(max_tasks=self.config.get('magpie_tasks', 100))
        self.magpie_tasks = magpie_loader.load_magpie_subset()
        
        # Create instruction dataset
        self.instruction_dataset = InstructionDataset(self.magpie_tasks, 
                                                      self.config['dataset_config'])
        
        logger.info("=" * 80)
        logger.info(f"Data preparation complete!")
        logger.info(f"  - Wikipedia documents: {len(self.wikipedia_docs)}")
        logger.info(f"  - Instruction tasks: {len(self.magpie_tasks)}")
        logger.info("=" * 80)
    
    def get_data_loaders(self):
        """Get train/valid/test splits"""
        train, valid, test = self.instruction_dataset.get_task_split()
        return {
            'train': train,
            'valid': valid,
            'test': test,
            'wikipedia': self.wikipedia_docs,
        }

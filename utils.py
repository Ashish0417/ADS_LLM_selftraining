# Utility functions for ADS framework
import os
import json
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import hashlib
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCache:
    """Efficient caching mechanism for CPU laptop"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
    
    def get_cache_path(self, key: str) -> Path:
        """Generate cache file path from key"""
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.pkl"
    
    def save(self, key: str, data: Any, ttl: Optional[int] = None):
        """Save data to cache with optional TTL"""
        path = self.get_cache_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            self.metadata[key] = {
                'timestamp': time.time(),
                'ttl': ttl,
                'path': str(path)
            }
            logger.debug(f"Cached: {key}")
        except Exception as e:
            logger.warning(f"Failed to cache {key}: {e}")
    
    def load(self, key: str) -> Optional[Any]:
        """Load data from cache if exists and not expired"""
        path = self.get_cache_path(key)
        if not path.exists():
            return None
        
        # Check TTL
        if key in self.metadata:
            meta = self.metadata[key]
            if meta['ttl'] is not None:
                elapsed = time.time() - meta['timestamp']
                if elapsed > meta['ttl']:
                    path.unlink()
                    del self.metadata[key]
                    return None
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded from cache: {key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {key}: {e}")
            return None
    
    def clear_all(self):
        """Clear all cache files"""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
        self.metadata.clear()
        logger.info("Cache cleared")


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 512) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def split_into_chunks(text: str, chunk_size: int = 512, 
                         stride: int = 128) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - stride):
            chunk = ' '.join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate of token count (avg 1 token per 4 chars)"""
        return len(text) // 4


class MetricsLogger:
    """Log and track metrics during training"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = []
    
    def log(self, **kwargs):
        """Log metrics"""
        entry = {
            'timestamp': time.time(),
            **kwargs
        }
        self.metrics.append(entry)
        logger.info(json.dumps(entry))
    
    def save(self):
        """Save metrics to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Metrics saved to {self.log_file}")
    
    def summary(self) -> Dict:
        """Get summary statistics"""
        if not self.metrics:
            return {}
        
        summary = {
            'total_entries': len(self.metrics),
            'duration_seconds': self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']
        }
        
        # Calculate averages for numeric fields
        numeric_fields = {}
        for metric in self.metrics:
            for key, value in metric.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    if key not in numeric_fields:
                        numeric_fields[key] = []
                    numeric_fields[key].append(value)
        
        for key, values in numeric_fields.items():
            summary[f'avg_{key}'] = np.mean(values)
            summary[f'max_{key}'] = np.max(values)
            summary[f'min_{key}'] = np.min(values)
        
        return summary


class ProgressTracker:
    """Track progress with time estimation"""
    
    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.current = 0
        self.name = name
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def update(self, count: int = 1):
        """Update progress"""
        self.current += count
        current_time = time.time()
        
        # Log every second
        if current_time - self.last_log_time > 1.0:
            elapsed = current_time - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            percentage = 100 * self.current / self.total
            print(f"\r{self.name}: {self.current}/{self.total} "
                  f"({percentage:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - "
                  f"ETA: {remaining:.1f}s", end='', flush=True)
            
            self.last_log_time = current_time
    
    def finish(self):
        """Mark as finished"""
        elapsed = time.time() - self.start_time
        print(f"\n{self.name} completed in {elapsed:.1f}s")


def ensure_directory(path: str) -> Path:
    """Ensure directory exists"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict, file_path: str):
    """Save dictionary as JSON"""
    ensure_directory(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved to {file_path}")


def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def batch_iterator(items: List[Any], batch_size: int):
    """Iterate over items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i+batch_size]


def calculate_rouge_l(reference: str, candidate: str) -> float:
    """Calculate ROUGE-L similarity (simplified)"""
    ref_words = set(reference.lower().split())
    cand_words = set(candidate.lower().split())
    
    if not ref_words and not cand_words:
        return 1.0
    
    intersection = len(ref_words & cand_words)
    union = len(ref_words | cand_words)
    
    return intersection / union if union > 0 else 0.0


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


class APITrajectoryParser:
    """Parse and validate API trajectories"""
    
    API_NAMES = ["information_retrieval", "demonstration_generation", 
                 "question_answering", "none"]
    
    @staticmethod
    def parse_trajectory(text: str) -> List[Dict]:
        """Parse API trajectory from model output"""
        trajectories = []
        
        # Look for API calls in XML format
        import re
        pattern = r'<api>(\w+)\((.*?)\)</api>'
        matches = re.findall(pattern, text)
        
        for api_name, params in matches:
            if api_name in APITrajectoryParser.API_NAMES:
                # Clean parameters
                param_str = params.strip().strip('"\'')
                trajectories.append({
                    'name': api_name,
                    'param': param_str
                })
        
        return trajectories if trajectories else [{'name': 'none', 'param': ''}]
    
    @staticmethod
    def trajectory_to_string(trajectories: List[Dict]) -> str:
        """Convert trajectory list to string format"""
        if not trajectories or trajectories[0]['name'] == 'none':
            return "<api calls>none</api calls>"
        
        api_str = "".join([
            f"<api>{t['name']}({t['param']})</api>" 
            for t in trajectories
        ])
        return f"<api calls>{api_str}</api calls>"


class HeuristicScorer:
    """Heuristic-based scoring for evaluation (no reward model)"""
    
    @staticmethod
    def score_response(instruction: str, response: str) -> float:
        """Score response using heuristics (0-1)"""
        score = 0.0
        
        # Length check: responses should be substantial
        if len(response.split()) > 20:
            score += 0.3
        elif len(response.split()) > 10:
            score += 0.15
        
        # Coverage check: many words from instruction
        inst_words = set(instruction.lower().split())
        resp_words = set(response.lower().split())
        coverage = len(inst_words & resp_words) / len(inst_words) if inst_words else 0
        score += coverage * 0.2
        
        # Diversity check: unique words
        unique_ratio = len(resp_words) / len(response.split()) if response.split() else 0
        if unique_ratio > 0.7:
            score += 0.3
        elif unique_ratio > 0.5:
            score += 0.15
        
        # Structure check: has punctuation and structure
        if '.' in response and len(response) > 50:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0

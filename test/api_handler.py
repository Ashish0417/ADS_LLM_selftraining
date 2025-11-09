import logging
import json
import wikipedia
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SEMANTIC = True
except ImportError:
    HAS_SEMANTIC = False
    logging.warning("sentence-transformers not installed. Semantic search disabled.")

logger = logging.getLogger(__name__)

# Cache file for online Wikipedia fetches
ONLINE_CACHE_FILE = "cache/wikipedia_online_cache.json"


class HybridRetrievalAPI:
    """
    Hybrid Information Retrieval API using BM25 + Semantic Search + Real-time Wikipedia
    
    Features:
    - BM25 keyword-based search on local documents
    - Semantic search using sentence embeddings
    - Real-time Wikipedia API fallback for unknown topics
    - Intelligent caching of online fetches
    """
    
    def __init__(self, documents: List[str], cache_dir: str = "cache", use_semantic: bool = True):
        """Initialize Hybrid Retrieval API"""
        self.documents = documents
        self.cache_dir = cache_dir
        self.use_semantic = use_semantic and HAS_SEMANTIC
        self.online_cache_file = Path(cache_dir) / "wikipedia_online_cache.json"
        
        logger.info("=" * 80)
        logger.info("INITIALIZING HYBRID RETRIEVAL API")
        logger.info("=" * 80)
        
        # Build BM25 index
        logger.info(f"Building BM25 index for {len(documents)} documents...")
        self._build_bm25_index()
        
        # Load semantic model if enabled
        if self.use_semantic:
            logger.info("Loading semantic model (sentence-transformers)...")
            self._init_semantic_search()
        else:
            logger.info("Semantic search disabled (install sentence-transformers to enable)")
        
        # Load online cache
        self.online_cache = self._load_online_cache()
        logger.info(f"Loaded {len(self.online_cache)} cached online articles")
        
        logger.info("=" * 80)
        logger.info("HYBRID RETRIEVAL API READY")
        logger.info("=" * 80 + "\n")
    
    def _build_bm25_index(self):
        """Build BM25 index from documents"""
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info(f"  ✓ BM25 index built")
    
    def _init_semantic_search(self):
        """Initialize semantic search with pre-trained model"""
        try:
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Encode all documents
            logger.info("  Encoding documents with semantic model (this may take a minute)...")
            self.doc_embeddings = self.semantic_model.encode(
                self.documents,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            logger.info(f"  ✓ Semantic embeddings computed")
        except Exception as e:
            logger.error(f"Failed to initialize semantic search: {e}")
            self.use_semantic = False
    
    def _load_online_cache(self) -> Dict[str, str]:
        """Load cache of online Wikipedia articles"""
        if self.online_cache_file.exists():
            try:
                with open(self.online_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load online cache: {e}")
        return {}
    
    def _save_online_cache(self):
        """Save online cache to file"""
        self.online_cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.online_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.online_cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save online cache: {e}")
    
    def _bm25_search(self, query: str, top_k: int = 5) -> tuple:
        """BM25 keyword-based search"""
        if not query or not query.strip():
            return [], []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [self.documents[i] for i in top_indices if scores[i] > 0]
        result_scores = [scores[i] for i in top_indices if scores[i] > 0]
        
        return results, result_scores
    
    def _semantic_search(self, query: str, top_k: int = 5) -> tuple:
        """Semantic similarity search using embeddings"""
        if not self.use_semantic or not query or not query.strip():
            return [], []
        
        try:
            query_embedding = self.semantic_model.encode(query, convert_to_tensor=True)
            hits = util.semantic_search(query_embedding, self.doc_embeddings, top_k=top_k)
            
            results = [self.documents[hit['corpus_id']] for hit in hits[0]]
            scores = [hit['score'] for hit in hits[0]]
            
            return results, scores
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return [], []
    
    def _fetch_from_wikipedia_online(self, query: str) -> str:
        """Fetch article from Wikipedia API in real-time"""
        try:
            logger.info(f"[WIKIPEDIA-ONLINE] Fetching: {query[:60]}...")
            
            # Search Wikipedia
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                logger.warning(f"[WIKIPEDIA-ONLINE] No results found for: {query}")
                return ""
            
            # Get first result
            title = search_results[0]
            try:
                page = wikipedia.page(title)
                article_text = page.content
                
                # Cache it
                cache_key = query.lower().strip()
                self.online_cache[cache_key] = article_text
                self._save_online_cache()
                
                logger.info(f"[WIKIPEDIA-ONLINE] ✓ Fetched and cached: {title}")
                return article_text
            
            except wikipedia.exceptions.DisambiguationError:
                logger.warning(f"[WIKIPEDIA-ONLINE] Disambiguation page for: {title}")
                return ""
            except wikipedia.exceptions.PageError:
                logger.warning(f"[WIKIPEDIA-ONLINE] Page not found: {title}")
                return ""
            except Exception as e:
                logger.warning(f"[WIKIPEDIA-ONLINE] Error fetching {title}: {e}")
                return ""
        
        except Exception as e:
            logger.error(f"[WIKIPEDIA-ONLINE] Error: {e}")
            return ""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Hybrid retrieval: BM25 + Semantic + Real-time Wikipedia fallback
        
        Process:
        1. Try local BM25 search
        2. Try local semantic search
        3. Combine and rank results
        4. If poor quality, fetch from Wikipedia API
        5. Cache online results
        """
        
        if not query or not query.strip():
            logger.warning("Empty query provided to IR API")
            return []
        
        logger.info(f"[HYBRID] Retrieving for: {query[:60]}...")
        
        # Step 1: BM25 Search
        logger.info("  [Step 1] BM25 search...")
        bm25_results, bm25_scores = self._bm25_search(query, top_k=top_k)
        bm25_avg_score = np.mean(bm25_scores) if bm25_scores else 0
        logger.info(f"    BM25: {len(bm25_results)} results (avg score: {bm25_avg_score:.3f})")
        
        # Step 2: Semantic Search
        logger.info("  [Step 2] Semantic search...")
        semantic_results, semantic_scores = self._semantic_search(query, top_k=top_k)
        semantic_avg_score = np.mean(semantic_scores) if semantic_scores else 0
        logger.info(f"    Semantic: {len(semantic_results)} results (avg score: {semantic_avg_score:.3f})")
        
        # Step 3: Combine results (prioritize semantic for better quality)
        combined_results = list(dict.fromkeys(semantic_results + bm25_results))[:top_k]
        
        # Step 4: Check if results are good quality
        has_good_results = (
            len(combined_results) > 0 and 
            any(len(doc) > 200 for doc in combined_results) and
            (bm25_avg_score > 0.5 or semantic_avg_score > 0.5)
        )
        
        if has_good_results:
            logger.info(f"  ✓ Found {len(combined_results)} relevant local results")
            return combined_results
        
        # Step 5: Fallback to real-time Wikipedia
        logger.info("  [Step 3] Local results insufficient, trying Wikipedia API...")
        
        # Check cache first
        cache_key = query.lower().strip()
        if cache_key in self.online_cache:
            logger.info(f"    ✓ Found in online cache")
            return [self.online_cache[cache_key]]
        
        # Fetch from Wikipedia
        online_result = self._fetch_from_wikipedia_online(query)
        if online_result:
            return [online_result]
        
        logger.warning(f"  ✗ No results found anywhere for: {query}")
        return []


class DemonstrationGenerationAPI:
    """Generate demonstration examples"""
    
    def __init__(self, generator_model, tokenizer, cache_dir: str = "cache"):
        """Initialize demonstration generation API"""
        self.generator_model = generator_model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        logger.info("Initialized Demonstration Generation API")
    
    def generate(self, instruction: str, num_samples: int = 3) -> List[Dict]:
        """Generate demonstration examples"""
        demonstrations = []
        
        for i in range(num_samples):
            try:
                inputs = self.tokenizer(
                    f"Generate example for: {instruction}",
                    return_tensors="pt",
                    max_length=128,
                    truncation=True
                )
                
                outputs = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7
                )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                demonstrations.append({
                    'input': instruction,
                    'output': response
                })
            except Exception as e:
                logger.warning(f"Failed to generate demo {i}: {e}")
                continue
        
        logger.info(f"Demo API generated {len(demonstrations)} examples")
        return demonstrations


class QuestionAnsweringAPI:
    """Question Answering API"""
    
    def __init__(self, qa_model, tokenizer, cache_dir: str = "cache"):
        """Initialize QA API"""
        self.qa_model = qa_model
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        logger.info("Initialized Question Answering API")
    
    def answer(self, question: str, context: str = "") -> str:
        """Answer question using model"""
        try:
            prompt = f"Question: {question}\n"
            if context:
                prompt = f"Context: {context}\n{prompt}"
            prompt += "Answer:"
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            )
            
            outputs = self.qa_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"QA API answered question: '{question[:50]}...'")
            
            return answer
        except Exception as e:
            logger.error(f"QA API error: {e}")
            return ""


class APIExecutor:
    """Execute API trajectories with hybrid retrieval"""
    
    def __init__(self, ir_api, demo_api, qa_api):
        """Initialize API executor"""
        self.ir_api = ir_api
        self.demo_api = demo_api
        self.qa_api = qa_api
        logger.info("Initialized APIExecutor")
    
    def execute_trajectory(self, trajectory, instruction: str = ""):
        """
        Execute API trajectory with hybrid search
        
        Returns:
            dict: {
                'collected_data': str (retrieved Wikipedia text),
                'total_cost': int,
                'api_calls': list
            }
        """
        
        if not instruction or not instruction.strip():
            logger.warning("Empty instruction provided to execute_trajectory")
            return {'collected_data': '', 'total_cost': 0, 'api_calls': []}
        
        try:
            logger.info(f"[APIExecutor] Query: {instruction[:70]}...")
            
            # Use hybrid retrieval (BM25 + Semantic + Online Fallback)
            results = self.ir_api.retrieve(instruction, top_k=5)
            
            if results:
                # Format top 3 results with some context
                collected_data = "\n\n".join(results[:3])
                logger.info(f"[APIExecutor] ✓ Retrieved {len(results)} results")
            else:
                collected_data = ""
                logger.warning(f"[APIExecutor] ✗ No results found")
            
            return {
                'collected_data': collected_data,
                'total_cost': 1,  # IR API costs 1 unit
                'api_calls': ['information_retrieval']
            }
        
        except Exception as e:
            logger.error(f"[APIExecutor] Error: {e}")
            return {'collected_data': '', 'total_cost': 0, 'api_calls': []}

# retrieval_engine.py
# INTEGRATED RETRIEVAL ENGINE - Advanced Multi-Stage Retrieval System
# Drop-in replacement for old retrieval system
# Efficient integration with your ADS framework

import logging
import wikipedia
import spacy
import requests
import time
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

logger = logging.getLogger(__name__)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)


class EntityLinker:
    """Stage 1: Entity Extraction & Linking via Wikidata"""
    
    def __init__(self):
        logger.info("[EntityLinker] Initializing...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("[EntityLinker] Downloading spaCy model...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self.wikidata_api = "https://www.wikidata.org/w/api.php"
        self.max_article_length = 5000
        logger.info("[EntityLinker] ✓ Ready")
    
    def extract_entities(self, text: str, min_length: int = 2) -> List[Dict]:
        """Extract named entities"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT"] and len(ent.text) >= min_length:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
        
        return entities
    
    def link_entity_to_wikipedia(self, entity_text: str, timeout: int = 10) -> Optional[str]:
        """Link entity to Wikipedia via Wikidata"""
        try:
            params = {
                'action': 'wbsearchentities',
                'search': entity_text,
                'language': 'en',
                'format': 'json'
            }
            
            headers = {'User-Agent': 'ADS-Retrieval-Engine/1.0'}
            response = requests.get(self.wikidata_api, params=params, timeout=timeout, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('search'):
                wikidata_id = data['search'][0]['id']
                
                wikidata_detail_params = {
                    'action': 'wbgetentities',
                    'ids': wikidata_id,
                    'props': 'sitelinks',
                    'format': 'json'
                }
                
                detail_response = requests.get(
                    self.wikidata_api, params=wikidata_detail_params, timeout=timeout, headers=headers
                )
                detail_response.raise_for_status()
                detail_data = detail_response.json()
                
                if wikidata_id in detail_data['entities']:
                    entity = detail_data['entities'][wikidata_id]
                    if 'sitelinks' in entity and 'enwiki' in entity['sitelinks']:
                        wiki_title = entity['sitelinks']['enwiki']['title']
                        
                        try:
                            page = wikipedia.page(wiki_title, auto_suggest=False)
                            content = page.content
                            
                            if len(content) > self.max_article_length:
                                content = content[:self.max_article_length]
                            
                            return content
                        except:
                            return None
        
        except Exception as e:
            logger.debug(f"[EntityLinker] Error linking '{entity_text}': {e}")
        
        return None
    
    def link_entities(self, text: str) -> List[Tuple[str, Optional[str]]]:
        """Extract and link entities to Wikipedia"""
        entities = self.extract_entities(text)
        results = []
        
        for entity in entities:
            try:
                wiki_article = self.link_entity_to_wikipedia(entity['text'])
                if wiki_article:
                    results.append((entity['text'], wiki_article))
            except:
                continue
        
        return results


class BM25Retriever:
    """Stage 2-3: BM25 Sparse Retrieval"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.max_article_length = 5000
        logger.info("[BM25Retriever] ✓ Ready")
    
    def search_wikipedia(self, query: str, top_k: int = 20) -> List[str]:
        """Search Wikipedia and return articles"""
        try:
            search_results = wikipedia.search(query, results=top_k)
            
            articles = []
            seen_articles = set()
            
            for title in search_results[:top_k]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    content = page.content
                    
                    article_id = hash(content[:200])
                    if article_id in seen_articles:
                        continue
                    seen_articles.add(article_id)
                    
                    if len(content) > self.max_article_length:
                        content = content[:self.max_article_length]
                    
                    articles.append(content)
                except:
                    continue
            
            return articles
        
        except Exception as e:
            logger.error(f"[BM25] Wikipedia search failed: {e}")
            return []


class DenseRanker:
    """Stage 4: Dense Re-ranking with Sentence-Transformers"""
    
    def __init__(self, batch_size: int = 32):
        if not HAS_SENTENCE_TRANSFORMERS:
            logger.warning("[DenseRanker] Sentence-transformers not available")
            self.model = None
            return
        
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.batch_size = batch_size
            logger.info("[DenseRanker] ✓ Ready")
        except Exception as e:
            logger.error(f"[DenseRanker] Failed to load: {e}")
            self.model = None
    
    def rank(self, query: str, candidates: List[str], top_k: int = 5) -> List[str]:
        """Rank candidates by semantic similarity"""
        if not self.model:
            return candidates[:top_k]
        
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            
            candidate_texts = [c[:500] for c in candidates]
            candidate_embeddings = self.model.encode(
                candidate_texts,
                convert_to_tensor=True,
                batch_size=self.batch_size
            )
            
            scores = util.semantic_search(query_embedding, candidate_embeddings, top_k=len(candidates))
            
            ranked_candidates = []
            for hit in scores[0][:top_k]:
                ranked_candidates.append(candidates[hit['corpus_id']])
            
            return ranked_candidates
        
        except Exception as e:
            logger.error(f"[DenseRanker] Failed: {e}")
            return candidates[:top_k]


class CrossEncoderRanker:
    """Stage 5: Cross-Encoder Final Ranking with BERT"""
    
    def __init__(self):
        if not HAS_CROSS_ENCODER:
            logger.warning("[CrossEncoder] Transformers/torch not available")
            self.model = None
            return
        
        try:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("[CrossEncoder] ✓ Ready")
        except Exception as e:
            logger.error(f"[CrossEncoder] Failed to load: {e}")
            self.model = None
    
    def rank(self, query: str, candidates: List[str], top_k: int = 3) -> List[str]:
        """Final ranking with cross-encoder"""
        if not self.model:
            return candidates[:top_k]
        
        try:
            scores = []
            
            for i, candidate in enumerate(candidates):
                candidate_short = candidate[:2000]
                
                inputs = self.tokenizer(
                    query, candidate_short,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    score = outputs.logits[0][0].item()
                
                scores.append((i, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            ranked_candidates = [candidates[idx] for idx, score in scores[:top_k]]
            
            return ranked_candidates
        
        except Exception as e:
            logger.error(f"[CrossEncoder] Failed: {e}")
            return candidates[:top_k]


class AdvancedRetrievalEngine:
    """
    MAIN RETRIEVAL ENGINE - Multi-Stage Pipeline
    Integrates: Entity Linking + BM25 + Dense + Cross-Encoder
    
    Drop-in replacement for old retrieval system
    """
    
    def __init__(self, use_cross_encoder: bool = True, use_dense: bool = True):
        """Initialize retrieval engine"""
        logger.info("=" * 80)
        logger.info("ADVANCED RETRIEVAL ENGINE - INTEGRATED")
        logger.info("=" * 80)
        
        self.entity_linker = EntityLinker()
        self.bm25_retriever = BM25Retriever()
        self.dense_ranker = DenseRanker() if use_dense else None
        self.cross_encoder = CrossEncoderRanker() if use_cross_encoder else None
        
        self.stats = {
            'total_queries': 0,
            'entity_linked': 0,
            'avg_time': 0
        }
        
        logger.info("✓ RETRIEVAL ENGINE READY")
        logger.info("=" * 80 + "\n")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Multi-stage retrieval
        Stage 1: Entity Linking
        Stage 2: BM25
        Stage 3: Dense Re-ranking
        Stage 4: Cross-Encoder
        """
        start_time = time.time()
        
        if not query or not query.strip():
            return []
        
        # Stage 1: Entity Linking
        entity_results = self.entity_linker.link_entities(query)
        
        if entity_results:
            self.stats['entity_linked'] += 1
            results = [article for _, article in entity_results[:top_k]]
            elapsed = time.time() - start_time
            self.stats['avg_time'] = (self.stats['avg_time'] * self.stats['total_queries'] + elapsed) / (self.stats['total_queries'] + 1)
            self.stats['total_queries'] += 1
            return results
        
        # Stage 2: BM25
        bm25_results = self.bm25_retriever.search_wikipedia(query, top_k=20)
        
        if not bm25_results:
            return []
        
        candidates = bm25_results
        
        # Stage 3: Dense Re-ranking
        if self.dense_ranker and len(candidates) > 5:
            candidates = self.dense_ranker.rank(query, candidates, top_k=5)
        
        # Stage 4: Cross-Encoder
        if self.cross_encoder and len(candidates) > top_k:
            candidates = self.cross_encoder.rank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]
        
        elapsed = time.time() - start_time
        self.stats['avg_time'] = (self.stats['avg_time'] * self.stats['total_queries'] + elapsed) / (self.stats['total_queries'] + 1)
        self.stats['total_queries'] += 1
        
        return candidates
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return self.stats

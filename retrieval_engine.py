
# ADVANCED WIKIPEDIA RETRIEVAL ENGINE
# Implements: Hybrid Search (BM25 + Dense Embeddings) + Entity Extraction + Cross-Encoder Reranking
# Based on research: Haystack, Qdrant, Weaviate, Elastic hybrid search architectures

import logging
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    import spacy
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False
    logger.warning("Advanced libraries not available. Install: sentence-transformers, rank-bm25, spacy")


class EntityExtractor:
    """
    Advanced Entity Extraction using SpaCy NER
    
    Extracts:
    - PERSON names (Prime Minister, President, Champion)
    - ORG organizations (Apple, Academy Award)
    - GPE geo-political entities (United Kingdom)
    - DATE dates and years (2025)
    - PRODUCT products (iPhone)
    - EVENT events (Formula 1 Championship)
    """
    
    def __init__(self):
        """Initialize entity extractor"""
        if ADVANCED_LIBS_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("[EntityExtractor] Loaded spaCy en_core_web_sm")
            except:
                logger.warning("[EntityExtractor] spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Returns:
            Dict with entity types as keys and lists of entities as values
        """
        if self.nlp is None:
            return self._fallback_extraction(text)
        
        try:
            doc = self.nlp(text)
            
            entities = defaultdict(list)
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
            
            return dict(entities)
        
        except Exception as e:
            logger.error(f"[EntityExtractor] Error: {e}")
            return self._fallback_extraction(text)
    
    def _fallback_extraction(self, text: str) -> Dict[str, List[str]]:
        """Fallback rule-based extraction"""
        entities = defaultdict(list)
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            entities['DATE'] = years
        
        # Extract capitalized phrases (potential names/places)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        if capitalized:
            entities['ENTITY'] = capitalized[:5]
        
        return dict(entities)
    
    def get_key_entities(self, text: str) -> List[str]:
        """Get most important entities for search"""
        entities_dict = self.extract_entities(text)
        
        # Priority order for entity types
        priority_types = ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'DATE']
        
        key_entities = []
        for entity_type in priority_types:
            if entity_type in entities_dict:
                key_entities.extend(entities_dict[entity_type][:3])  # Top 3 per type
        
        return key_entities[:6]  # Max 6 total


class HybridBM25DenseRetriever:
    """
    Hybrid Retrieval: BM25 (Keyword) + Dense Embeddings (Semantic)
    
    Architecture:
    1. BM25: Fast keyword matching (filters to top 100)
    2. Dense: Semantic understanding (refines to top 20)
    3. Cross-Encoder: Final reranking (selects top k)
    
    Based on: Haystack, Qdrant, Weaviate architectures
    """
    
    def __init__(
        self,
        dense_model_name: str = "BAAI/bge-small-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_model_name: Dense embedding model (BGE, MiniLM, etc.)
            cross_encoder_model: Reranker model
            use_cross_encoder: Whether to use cross-encoder reranking
        """
        self.use_cross_encoder = use_cross_encoder and ADVANCED_LIBS_AVAILABLE
        
        if ADVANCED_LIBS_AVAILABLE:
            try:
                # Load dense embedding model
                self.dense_model = SentenceTransformer(dense_model_name)
                logger.info(f"[HybridRetriever] Loaded dense model: {dense_model_name}")
                
                # Load cross-encoder reranker
                if self.use_cross_encoder:
                    self.cross_encoder = CrossEncoder(cross_encoder_model)
                    logger.info(f"[HybridRetriever] Loaded cross-encoder: {cross_encoder_model}")
            except Exception as e:
                logger.error(f"[HybridRetriever] Error loading models: {e}")
                self.dense_model = None
                self.cross_encoder = None
        else:
            self.dense_model = None
            self.cross_encoder = None
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        self.document_embeddings = None
        
        # Stats
        self.stats = {
            'total_queries': 0,
            'bm25_filtered': 0,
            'dense_refined': 0,
            'cross_encoder_reranked': 0
        }
    
    def index_documents(self, documents: List[str]):
        """
        Index documents for hybrid search
        
        Args:
            documents: List of document strings
        """
        logger.info(f"[HybridRetriever] Indexing {len(documents)} documents...")
        
        self.documents = documents
        
        # 1. Build BM25 index
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logger.info("[HybridRetriever] ✓ BM25 index built")
        
        # 2. Build dense embeddings
        if self.dense_model:
            logger.info("[HybridRetriever] Computing dense embeddings...")
            self.document_embeddings = self.dense_model.encode(
                documents,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info("[HybridRetriever] ✓ Dense embeddings computed")
        
        logger.info("[HybridRetriever] ✓ Indexing complete")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        bm25_top_k: int = 100,
        dense_top_k: int = 20
    ) -> List[str]:
        """
        Hybrid retrieval with 3-stage pipeline
        
        Stage 1: BM25 keyword filtering (top 100)
        Stage 2: Dense semantic refinement (top 20)
        Stage 3: Cross-encoder reranking (top k)
        
        Args:
            query: Search query
            top_k: Final number of results
            bm25_top_k: BM25 candidates
            dense_top_k: Dense refinement candidates
        
        Returns:
            List of top-k most relevant documents
        """
        self.stats['total_queries'] += 1
        
        if not self.documents:
            logger.warning("[HybridRetriever] No documents indexed!")
            return []
        
        logger.info(f"[HybridRetriever] Query: {query}")
        
        # Stage 1: BM25 Keyword Filtering
        bm25_candidates = self._bm25_retrieve(query, bm25_top_k)
        self.stats['bm25_filtered'] = len(bm25_candidates)
        logger.info(f"[HybridRetriever] BM25 filtered: {len(bm25_candidates)} docs")
        
        if not bm25_candidates:
            return []
        
        # Stage 2: Dense Semantic Refinement
        if self.dense_model and len(bm25_candidates) > dense_top_k:
            dense_candidates = self._dense_refine(query, bm25_candidates, dense_top_k)
            self.stats['dense_refined'] = len(dense_candidates)
            logger.info(f"[HybridRetriever] Dense refined: {len(dense_candidates)} docs")
        else:
            dense_candidates = bm25_candidates[:dense_top_k]
        
        # Stage 3: Cross-Encoder Reranking
        if self.use_cross_encoder and self.cross_encoder and len(dense_candidates) > top_k:
            final_results = self._cross_encoder_rerank(query, dense_candidates, top_k)
            self.stats['cross_encoder_reranked'] = len(final_results)
            logger.info(f"[HybridRetriever] Cross-encoder reranked: {top_k} docs")
        else:
            final_results = dense_candidates[:top_k]
        
        logger.info(f"[HybridRetriever] ✓ Retrieved {len(final_results)} documents")
        
        return final_results
    
    def _bm25_retrieve(self, query: str, top_k: int) -> List[str]:
        """
        Stage 1: BM25 keyword-based retrieval
        
        BM25 Formula:
        score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D| / avgdl))
        
        Where:
        - f(qi,D) = term frequency of qi in document D
        - |D| = document length
        - avgdl = average document length
        - k1 = term frequency saturation (default: 1.2)
        - b = length normalization (default: 0.75)
        """
        if not self.bm25:
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Filter out zero scores
        candidates = [
            self.documents[idx]
            for idx in top_indices
            if scores[idx] > 0
        ]
        
        return candidates
    
    def _dense_refine(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """
        Stage 2: Dense embedding refinement
        
        Uses cosine similarity between query and document embeddings
        """
        # Encode query
        query_embedding = self.dense_model.encode(query, convert_to_numpy=True)
        
        # Encode candidates
        candidate_embeddings = self.dense_model.encode(candidates, convert_to_numpy=True)
        
        # Compute cosine similarities
        similarities = np.dot(candidate_embeddings, query_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        refined = [candidates[idx] for idx in top_indices]
        
        return refined
    
    def _cross_encoder_rerank(self, query: str, candidates: List[str], top_k: int) -> List[str]:
        """
        Stage 3: Cross-encoder reranking
        
        Cross-encoder computes direct query-document relevance score
        More accurate than bi-encoder but slower
        """
        # Prepare pairs
        pairs = [[query, doc] for doc in candidates]
        
        # Compute scores
        scores = self.cross_encoder.predict(pairs)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        reranked = [candidates[idx] for idx in top_indices]
        
        return reranked
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        # Simple tokenization (can be improved with lemmatization)
        text_lower = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text_lower)
        return tokens
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return self.stats.copy()


class AdvancedRetrievalEngine:
    """
    STATE-OF-THE-ART Wikipedia Retrieval Engine
    
    Features:
    1. Entity Extraction (SpaCy NER)
    2. Query Enhancement (entity-based expansion)
    3. Hybrid Retrieval (BM25 + Dense + Cross-Encoder)
    4. Multi-stage Ranking
    5. Relevance Filtering
    
    Architecture inspired by: Haystack, Qdrant, Weaviate, Elastic
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = True,
        use_dense: bool = True,
        dense_model: str = "BAAI/bge-small-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize advanced retrieval engine
        
        Args:
            use_cross_encoder: Enable cross-encoder reranking
            use_dense: Enable dense embeddings
            dense_model: Dense embedding model name
            cross_encoder_model: Cross-encoder model name
        """
        logger.info("=" * 80)
        logger.info("INITIALIZING ADVANCED RETRIEVAL ENGINE")
        logger.info("=" * 80)
        
        # Initialize components
        self.entity_extractor = EntityExtractor()
        
        self.hybrid_retriever = HybridBM25DenseRetriever(
            dense_model_name=dense_model,
            cross_encoder_model=cross_encoder_model,
            use_cross_encoder=use_cross_encoder
        )
        
        self.use_dense = use_dense and ADVANCED_LIBS_AVAILABLE
        self.use_cross_encoder = use_cross_encoder and ADVANCED_LIBS_AVAILABLE
        
        # Cache
        self.query_cache = {}
        
        # Stats
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'entity_extraction_success': 0,
            'avg_retrieval_time': 0.0
        }
        
        logger.info(f"Dense embeddings: {'✓' if self.use_dense else '✗'}")
        logger.info(f"Cross-encoder: {'✓' if self.use_cross_encoder else '✗'}")
        logger.info(f"Entity extraction: {'✓' if self.entity_extractor.nlp else '✗'}")
        logger.info("=" * 80)
    
    def index_documents(self, documents: List[str]):
        """Index Wikipedia documents"""
        logger.info(f"[AdvancedRetrieval] Indexing {len(documents)} documents...")
        self.hybrid_retriever.index_documents(documents)
        logger.info("[AdvancedRetrieval] ✓ Indexing complete")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Advanced multi-stage retrieval
        
        Pipeline:
        1. Entity Extraction → Identify key entities in query
        2. Query Enhancement → Build better search query
        3. Hybrid Retrieval → BM25 + Dense + Cross-Encoder
        4. Relevance Filtering → Final validation
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of top-k most relevant documents
        """
        import time
        start_time = time.time()
        
        self.stats['total_queries'] += 1
        
        # Check cache
        cache_key = f"{query}_{top_k}"
        if cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            logger.info("[AdvancedRetrieval] Cache hit!")
            return self.query_cache[cache_key]
        
        logger.info(f"\n[AdvancedRetrieval] Query: {query}")
        
        # Step 1: Entity Extraction
        entities = self.entity_extractor.get_key_entities(query)
        if entities:
            self.stats['entity_extraction_success'] += 1
            logger.info(f"[AdvancedRetrieval] Entities: {entities}")
        
        # Step 2: Query Enhancement
        enhanced_query = self._enhance_query(query, entities)
        logger.info(f"[AdvancedRetrieval] Enhanced query: {enhanced_query}")
        
        # Step 3: Hybrid Retrieval
        results = self.hybrid_retriever.retrieve(
            enhanced_query,
            top_k=top_k,
            bm25_top_k=100,
            dense_top_k=20
        )
        
        # Step 4: Relevance Filtering
        filtered_results = self._filter_by_relevance(query, entities, results)
        
        # Update stats
        retrieval_time = time.time() - start_time
        self.stats['avg_retrieval_time'] = (
            (self.stats['avg_retrieval_time'] * (self.stats['total_queries'] - 1) + retrieval_time)
            / self.stats['total_queries']
        )
        
        # Cache results
        self.query_cache[cache_key] = filtered_results
        
        logger.info(f"[AdvancedRetrieval] ✓ Retrieved {len(filtered_results)} docs in {retrieval_time:.2f}s")
        
        return filtered_results
    
    def _enhance_query(self, query: str, entities: List[str]) -> str:
        """
        Enhance query with extracted entities
        
        Example:
        Query: "Who is current UK PM 2025?"
        Entities: ["Prime Minister", "United Kingdom", "2025"]
        Enhanced: "Prime Minister United Kingdom 2025 current"
        """
        # Add entity keywords
        enhanced = query
        
        # Add important keywords
        for entity in entities:
            if entity.lower() not in query.lower():
                enhanced += f" {entity}"
        
        # Add domain-specific expansions
        if "prime minister" in query.lower():
            enhanced += " PM government leader"
        
        if "best picture" in query.lower():
            enhanced += " Academy Award Oscar film movie"
        
        if "champion" in query.lower() and ("formula" in query.lower() or "f1" in query.lower()):
            enhanced += " F1 Formula One racing winner"
        
        if "iphone" in query.lower():
            enhanced += " Apple smartphone mobile device"
        
        return enhanced
    
    def _filter_by_relevance(
        self,
        query: str,
        entities: List[str],
        candidates: List[str],
        min_entity_match: float = 0.3
    ) -> List[str]:
        """
        Filter candidates by relevance to query entities
        
        Args:
            query: Original query
            entities: Extracted entities
            candidates: Candidate documents
            min_entity_match: Minimum entity match ratio
        
        Returns:
            Filtered documents
        """
        if not entities:
            return candidates
        
        filtered = []
        for doc in candidates:
            # Count entity matches
            doc_lower = doc.lower()
            matches = sum(1 for e in entities if e.lower() in doc_lower)
            match_ratio = matches / len(entities)
            
            if match_ratio >= min_entity_match:
                filtered.append(doc)
            else:
                logger.debug(f"[AdvancedRetrieval] Filtered out (low entity match): {doc[:60]}...")
        
        return filtered if filtered else candidates  # Return all if none match
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        stats = self.stats.copy()
        stats.update(self.hybrid_retriever.get_stats())
        return stats
    
    def print_stats(self):
        """Print retrieval statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 80)
        print("RETRIEVAL ENGINE STATISTICS")
        print("=" * 80)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Cache hits: {stats['cache_hits']} ({stats['cache_hits']/max(stats['total_queries'],1)*100:.1f}%)")
        print(f"Entity extraction success: {stats['entity_extraction_success']}")
        print(f"Avg retrieval time: {stats['avg_retrieval_time']:.2f}s")
        print(f"Avg BM25 candidates: {stats['bm25_filtered']}")
        print(f"Avg dense refined: {stats['dense_refined']}")
        print(f"Avg cross-encoder reranked: {stats['cross_encoder_reranked']}")
        print("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    # Initialize engine
    engine = AdvancedRetrievalEngine(
        use_cross_encoder=True,
        use_dense=True
    )
    
    # Example documents (simplified)
    docs = [
        "The United Kingdom of Great Britain and Northern Ireland is a country...",
        "Keir Starmer became Prime Minister of the United Kingdom in 2024...",
        "The Academy Award for Best Picture is one of the Academy Awards...",
        "Formula One World Championship 2025 season features top drivers...",
        "iPhone 16 was released by Apple in September 2025..."
    ]
    
    engine.index_documents(docs)
    
    # Test queries
    test_queries = [
        "Who is current UK Prime Minister 2025?",
        "Which movie won Best Picture 2025?",
        "Who is F1 Champion 2025?",
        "What is latest iPhone 2025?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = engine.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result[:100]}...")
    
    # Print stats
    engine.print_stats()

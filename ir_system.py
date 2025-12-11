"""
Information Retrieval System using SPLADE + BM25 Hybrid with RM3 Query Expansion

This IR system implements the best performing configuration from experiments:
- SPLADE (naver/splade-cocondenser-ensembledistil) for neural sparse retrieval
- BM25 (k1=0.9, b=0.4) with RM3 query expansion for lexical retrieval
- Hybrid fusion with weight 20 for BM25+RM3 component

Usage:
    from ir_system import IRSystem
    
    # Initialize the system
    ir = IRSystem()
    
    # Search for documents
    results = ir.search("What is information retrieval?")
    
    # Search with custom number of results
    results = ir.search("machine learning applications", top_k=20)
"""

import os
import warnings
from typing import Optional, List, Dict, Any

import pandas as pd
import pyt_splade
import pyterrier as pt
from pyterrier.measures import MAP, nDCG, Recall
import torch

warnings.filterwarnings("ignore", message="User provided device_type of 'cuda'")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")


def detect_device() -> str:
    """Detect the best available device for computation."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class IRSystem:
    """
    Hybrid Information Retrieval System combining SPLADE and BM25 with RM3 expansion.
    
    This system uses the best configuration found through extensive experimentation:
    - SPLADE for neural sparse retrieval
    - Tuned BM25 (k1=0.9, b=0.4) with RM3 query expansion
    - Hybrid fusion: SPLADE + 20 * BM25_RM3
    
    Attributes:
        device (str): The computation device (cuda, mps, or cpu)
        dataset: The PyTerrier dataset
        retriever: The hybrid retrieval pipeline
    """
    
    # Best configuration from experiments
    BM25_K1 = 0.9
    BM25_B = 0.4
    RM3_FB_DOCS = 10
    RM3_FB_TERMS = 15
    RM3_FB_LAMBDA = 0.5
    HYBRID_WEIGHT = 20
    
    def __init__(
        self,
        bm25_index_dir: str = "./robust04_bm25_index",
        splade_index_dir: str = "./robust04_splade_index",
        splade_model: str = "naver/splade-cocondenser-ensembledistil",
        device: Optional[str] = None,
        dataset_name: str = "irds:disks45/nocr/trec-robust-2004"
    ):
        """
        Initialize the IR system.
        
        Args:
            bm25_index_dir: Directory for BM25 index
            splade_index_dir: Directory for SPLADE index
            splade_model: SPLADE model name or path
            device: Computation device (auto-detected if None)
            dataset_name: PyTerrier dataset name
        """
        self.device = device or detect_device()
        print(f"Using device: {self.device}")
        
        # Initialize PyTerrier if not already done
        if not pt.started():
            pt.init()
        
        # Load dataset
        print("Loading dataset...")
        self.dataset = pt.get_dataset(dataset_name)
        
        # Initialize SPLADE model
        print("Initializing SPLADE model...")
        self.splade = pyt_splade.Splade(
            model=splade_model,
            device=self.device,
            max_length=256
        )
        
        # Setup indices
        self.bm25_index_dir = os.path.abspath(bm25_index_dir)
        self.splade_index_dir = os.path.abspath(splade_index_dir)
        
        # Load or create indices
        self._setup_indices()
        
        # Build retrieval pipeline
        self._build_retriever()
        
        print("IR System initialized successfully!")
    
    def _custom_corpus_iter(self):
        """Create custom corpus iterator combining title and body."""
        for doc in self.dataset.get_corpus_iter():
            yield {
                'docno': doc['docno'],
                'text': (doc.get('title', '') + ' ' + doc.get('body', '')).strip()
            }
    
    def _setup_indices(self):
        """Load existing indices or create new ones."""
        # BM25 Index
        if os.path.exists(self.bm25_index_dir):
            print("Loading existing BM25 index...")
            self.bm25_index_ref = pt.IndexFactory.of(self.bm25_index_dir)
        else:
            print("Creating BM25 index (this may take a while)...")
            bm25_indexer = pt.IterDictIndexer(self.bm25_index_dir)
            self.bm25_index_ref = bm25_indexer.index(self._custom_corpus_iter())
        
        # SPLADE Index
        if os.path.exists(self.splade_index_dir):
            print("Loading existing SPLADE index...")
            self.splade_index_ref = pt.IndexFactory.of(self.splade_index_dir)
        else:
            print("Creating SPLADE index (this may take a while)...")
            splade_indexer = self.splade.doc_encoder() >> pt.IterDictIndexer(self.splade_index_dir)
            self.splade_index_ref = splade_indexer.index(self._custom_corpus_iter())
    
    def _build_retriever(self):
        """Build the hybrid retrieval pipeline."""
        print("Building retrieval pipeline...")
        
        # SPLADE retriever
        self.splade_retriever = (
            self.splade.query_encoder() 
            >> pt.terrier.Retriever(self.splade_index_ref, wmodel="Tf")
        )
        
        # Tuned BM25 retriever
        self.bm25_retriever = pt.terrier.Retriever(
            self.bm25_index_ref,
            wmodel="BM25",
            controls={"bm25.k_1": self.BM25_K1, "bm25.b": self.BM25_B}
        )
        
        # BM25 with RM3 query expansion
        self.bm25_rm3_retriever = (
            self.bm25_retriever
            >> pt.text.get_text(self.dataset, ["title", "body"])
            >> pt.apply.text(lambda row: (row['title'] or '') + " " + (row['body'] or ''))
            >> pt.rewrite.RM3(
                self.bm25_index_ref,
                fb_docs=self.RM3_FB_DOCS,
                fb_terms=self.RM3_FB_TERMS,
                fb_lambda=self.RM3_FB_LAMBDA
            )
            >> self.bm25_retriever
        )
        
        # Hybrid retriever: SPLADE + weighted BM25+RM3
        self.retriever = self.splade_retriever + (self.HYBRID_WEIGHT * self.bm25_rm3_retriever)
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_text: bool = True
    ) -> pd.DataFrame:
        """
        Search for documents matching the query.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            return_text: Whether to include document text in results
            
        Returns:
            DataFrame with columns: qid, docno, score, rank, [title, body if return_text]
        """
        # Create query DataFrame
        query_df = pd.DataFrame([{"qid": "q1", "query": query}])
        
        # Retrieve results
        results = self.retriever.transform(query_df)
        
        # Limit to top_k
        results = results.head(top_k)
        
        # Add document text if requested
        if return_text and len(results) > 0:
            results = pt.text.get_text(self.dataset, ["title", "body"]).transform(results)
        
        return results
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        return_text: bool = False
    ) -> pd.DataFrame:
        """
        Search for multiple queries at once.
        
        Args:
            queries: List of query strings
            top_k: Number of top results per query
            return_text: Whether to include document text in results
            
        Returns:
            DataFrame with results for all queries
        """
        # Create query DataFrame
        query_df = pd.DataFrame([
            {"qid": f"q{i}", "query": q} 
            for i, q in enumerate(queries)
        ])
        
        # Retrieve results
        results = self.retriever.transform(query_df)
        
        # Limit to top_k per query
        results = results.groupby('qid').head(top_k).reset_index(drop=True)
        
        # Add document text if requested
        if return_text and len(results) > 0:
            results = pt.text.get_text(self.dataset, ["title", "body"]).transform(results)
        
        return results
    
    def get_document(self, docno: str) -> Dict[str, Any]:
        """
        Retrieve a specific document by its ID.
        
        Args:
            docno: The document number/ID
            
        Returns:
            Dictionary with document fields (docno, title, body)
        """
        doc_df = pd.DataFrame([{"docno": docno}])
        result = pt.text.get_text(self.dataset, ["title", "body"]).transform(doc_df)
        
        if len(result) > 0:
            row = result.iloc[0]
            return {
                "docno": row["docno"],
                "title": row.get("title", ""),
                "body": row.get("body", "")
            }
        return None
    
    def evaluate(self, topics: pd.DataFrame = None, qrels: pd.DataFrame = None) -> pd.DataFrame:
        """
        Evaluate the retrieval system on standard test collection.
        
        Args:
            topics: Query topics (uses dataset topics if None)
            qrels: Relevance judgments (uses dataset qrels if None)
            
        Returns:
            DataFrame with evaluation metrics
        """
        if topics is None:
            topics = self.dataset.get_topics()
            topics["query"] = topics["title"]
        
        if qrels is None:
            qrels = self.dataset.get_qrels()
        
        results = pt.Experiment(
            [self.splade_retriever, self.bm25_retriever, self.retriever],
            topics,
            qrels,
            eval_metrics=[MAP, nDCG @ 10, Recall @ 100],
            names=["SPLADE", "BM25", "Hybrid (Best)"]
        )
        
        return results


def interactive_search():
    """Run an interactive search session."""
    print("=" * 60)
    print("Interactive IR System")
    print("=" * 60)
    print()
    
    # Initialize system
    ir = IRSystem()
    
    print()
    print("Enter your search queries (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        query = input("\nQuery: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("\nSearching...")
        results = ir.search(query, top_k=10)
        
        if len(results) == 0:
            print("No results found.")
            continue
        
        print(f"\nTop {len(results)} results:")
        print("-" * 60)
        
        for i, row in results.iterrows():
            rank = row.get('rank', i + 1)
            docno = row['docno']
            score = row['score']
            title = row.get('title', 'N/A')
            
            # Truncate title if too long
            if title and len(title) > 80:
                title = title[:77] + "..."
            
            print(f"{rank}. [{docno}] (score: {score:.4f})")
            print(f"   Title: {title}")
            print()


if __name__ == "__main__":
    interactive_search()

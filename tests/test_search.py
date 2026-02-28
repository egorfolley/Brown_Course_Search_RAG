import json
from pathlib import Path
import pytest
from rag.vector_store import VectorStore
from rag.search import HybridSearch


@pytest.fixture
def sample_courses():
    """Load sample courses for testing."""
    return [
        {
            "course_code": "CSCI0320",
            "title": "Introduction to Software Engineering",
            "department": "Computer Science",
            "description": "Learn software engineering principles and practices.",
            "instructor": "John Doe",
            "meeting_times": "MWF 10:00-11:00",
            "prerequisites": "CSCI 0111",
            "source": "Bulletin",
        },
        {
            "course_code": "CSCI1420",
            "title": "Machine Learning",
            "department": "Computer Science",
            "description": "Introduction to machine learning algorithms and neural networks.",
            "instructor": "Jane Smith",
            "meeting_times": "TTh 13:00-14:30",
            "prerequisites": "CSCI 0320, MATH 0520",
            "source": "Bulletin",
        },
        {
            "course_code": "MATH0520",
            "title": "Linear Algebra",
            "department": "Mathematics",
            "description": "Fundamentals of linear algebra and matrix operations.",
            "instructor": "Bob Johnson",
            "meeting_times": "MWF 11:00-12:00",
            "prerequisites": "MATH 0100",
            "source": "Bulletin",
        },
    ]


@pytest.fixture
def vector_store(sample_courses, tmp_path):
    """Create a test vector store."""
    from rag.embedder import run as run_embedder
    
    # Generate embeddings for sample courses
    embeddings, _ = run_embedder(courses=sample_courses)
    
    # Create and save store
    store = VectorStore.build(embeddings, sample_courses)
    store.save(path=tmp_path)
    
    return VectorStore.load(path=tmp_path)


class TestVectorStore:
    """Test FAISS vector store functionality."""
    
    def test_vector_store_creation(self, vector_store):
        """Test that vector store is created correctly."""
        assert vector_store is not None
        assert len(vector_store.courses) == 3
    
    def test_semantic_search(self, vector_store):
        """Test semantic search returns results."""
        results = vector_store.search_semantic("machine learning", k=2)
        assert len(results) <= 2
        assert len(results) > 0
    
    def test_filtered_search(self, vector_store):
        """Test department filtering."""
        results = vector_store.search_semantic(
            "programming", 
            k=5, 
            filters={"department": "Computer Science"}
        )
        # All results should be from CS department
        for course in results:
            assert course["department"] == "Computer Science"


class TestHybridSearch:
    """Test hybrid search (FAISS + BM25)."""
    
    def test_hybrid_search_exact_match(self, sample_courses, vector_store):
        """Test that exact course code match is found."""
        search = HybridSearch(vector_store)
        
        results = search.query("CSCI0320", top_k=5)
        assert len(results) > 0
        
        # CSCI0320 should be in top results
        codes = [c["course_code"] for c in results]
        assert "CSCI0320" in codes
    
    def test_hybrid_search_semantic(self, sample_courses, vector_store):
        """Test semantic search for conceptual queries."""
        search = HybridSearch(vector_store)
        
        results = search.query("learning algorithms", top_k=5)
        assert len(results) > 0
        
        # Machine learning course should rank high
        codes = [c["course_code"] for c in results]
        assert "CSCI1420" in codes
    
    def test_hybrid_search_with_filter(self, sample_courses, vector_store):
        """Test hybrid search with department filter."""
        search = HybridSearch(vector_store)
        
        results = search.query("math linear", top_k=5, filters={"department": "Mathematics"})
        
        # All results should be from Mathematics
        for course in results:
            assert course["department"] == "Mathematics"
    
    def test_hybrid_search_returns_scores(self, sample_courses, vector_store):
        """Test that hybrid search returns similarity scores."""
        search = HybridSearch(vector_store)
        
        results = search.query("programming", top_k=3)
        
        # Each result should have _hybrid_score
        for course in results:
            assert "_hybrid_score" in course
            assert 0 <= course["_hybrid_score"] <= 1


class TestSearchIntegration:
    """Integration tests for full retrieval pipeline."""
    
    def test_end_to_end_query(self, sample_courses, vector_store):
        """Test complete query pipeline."""
        search = HybridSearch(vector_store)
        
        # Test various query types
        test_queries = [
            ("CSCI0320", None),  # Exact code match
            ("machine learning", None),  # Semantic query
            ("linear algebra", "Mathematics"),  # With filter
            ("software engineering friday", None),  # Multi-term
        ]
        
        for query, department in test_queries:
            filters = {"department": department} if department else None
            results = search.query(query, top_k=5, filters=filters)
            
            assert len(results) > 0, f"Query '{query}' returned no results"
            assert all("course_code" in r for r in results)
            assert all("_hybrid_score" in r for r in results)

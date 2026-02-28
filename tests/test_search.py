import json
from pathlib import Path
import pytest
from rag.vector_store import VectorStore
from rag.search import HybridSearch, extract_course_code, normalize_course_code


class TestCourseCodeExtraction:
    """Test course code extraction from queries."""
    
    def test_extract_uppercase_code(self):
        """Test extraction of uppercase course code."""
        code = extract_course_code("Who teaches ENGN0030?")
        assert code == "ENGN0030"
    
    def test_extract_lowercase_code(self):
        """Test extraction of lowercase course code."""
        code = extract_course_code("Find engn0030 on Fridays")
        assert code == "ENGN0030"
    
    def test_extract_code_with_space(self):
        """Test extraction of course code with space."""
        code = extract_course_code("What about ENGN 0030")
        assert code == "ENGN0030"
    
    def test_extract_amst_code(self):
        """Test extraction of AMST code."""
        code = extract_course_code("Find AMST2920 in American Studies")
        assert code == "AMST2920"
    
    def test_extract_csci_code(self):
        """Test extraction of CSCI code."""
        code = extract_course_code("CSCI0320 is a great course")
        assert code == "CSCI0320"
    
    def test_no_code_found(self):
        """Test when no course code in query."""
        code = extract_course_code("machine learning courses on Friday")
        assert code is None
    
    def test_multiple_codes_returns_first(self):
        """Test that first code is returned when multiple present."""
        code = extract_course_code("Compare CSCI0320 with CSCI1420")
        assert code == "CSCI0320"


class TestCourseCodeNormalization:
    """Test course code normalization."""
    
    def test_normalize_uppercase(self):
        """Test normalization preserves uppercase."""
        normalized = normalize_course_code("CSCI0320")
        assert normalized == "CSCI0320"
    
    def test_normalize_lowercase(self):
        """Test normalization converts to uppercase."""
        normalized = normalize_course_code("csci0320")
        assert normalized == "CSCI0320"
    
    def test_normalize_with_space(self):
        """Test normalization removes spaces."""
        normalized = normalize_course_code("CSCI 0320")
        assert normalized == "CSCI0320"
    
    def test_normalize_mixed_case(self):
        """Test normalization with mixed case."""
        normalized = normalize_course_code("CsCi 0320")
        assert normalized == "CSCI0320"


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
        {
            "course_code": "ENGN0030",
            "title": "Introduction to Engineering",
            "department": "The School of Engineering",
            "description": "Introduction to the profession of engineering.",
            "instructor": "D. Mittleman",
            "meeting_times": "MWF 9:00-9:50",
            "prerequisites": "",
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
        assert len(vector_store.courses) == 4
    
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
    """Test hybrid search (FAISS + BM25) with exact code matching."""
    
    def test_hybrid_search_returns_tuple(self, sample_courses, vector_store):
        """Test that hybrid search returns (results, detected_code) tuple."""
        search = HybridSearch(vector_store)
        
        result = search.query("CSCI0320", top_k=5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        results, detected_code = result
        assert isinstance(results, list)
        assert isinstance(detected_code, (str, type(None)))
    
    def test_hybrid_search_detects_code(self, sample_courses, vector_store):
        """Test that course code is detected in query."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("Who teaches ENGN0030?", top_k=5)
        assert detected_code == "ENGN0030"
    
    def test_exact_match_ranks_first(self, sample_courses, vector_store):
        """Test that exact code match ranks first."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("CSCI0320", top_k=5)
        assert len(results) > 0
        assert detected_code == "CSCI0320"
        
        # Exact match should rank first
        first_result = results[0]
        assert first_result["_exact_match"] is True
        assert first_result["course_code"] == "CSCI0320"
        assert first_result["_hybrid_score"] == 1.0
    
    def test_exact_match_with_instructor(self, sample_courses, vector_store):
        """Test that exact match includes instructor data."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("Who teaches ENGN0030, and when does the class meet?", top_k=5)
        assert detected_code == "ENGN0030"
        assert len(results) > 0
        
        # Find ENGN0030 in results
        engn_course = next((c for c in results if c["course_code"] == "ENGN0030"), None)
        assert engn_course is not None
        assert engn_course.get("instructor") == "D. Mittleman"
        assert engn_course.get("meeting_times") == "MWF 9:00-9:50"
    
    def test_hybrid_search_semantic(self, sample_courses, vector_store):
        """Test semantic search for conceptual queries."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("learning algorithms", top_k=5)
        assert len(results) > 0
        assert detected_code is None
        
        # Machine learning course should rank high
        codes = [c["course_code"] for c in results]
        assert "CSCI1420" in codes
    
    def test_hybrid_search_with_filter(self, sample_courses, vector_store):
        """Test hybrid search with department filter."""
        search = HybridSearch(vector_store)
        
        results, _ = search.query("math linear", top_k=5, filters={"department": "Mathematics"})
        
        # All results should be from Mathematics
        for course in results:
            assert course["department"] == "Mathematics"
    
    def test_hybrid_search_returns_scores(self, sample_courses, vector_store):
        """Test that hybrid search returns similarity scores."""
        search = HybridSearch(vector_store)
        
        results, _ = search.query("programming", top_k=3)
        
        # Each result should have _hybrid_score
        for course in results:
            assert "_hybrid_score" in course
            assert 0 <= course["_hybrid_score"] <= 1
    
    def test_exact_match_marked_in_results(self, sample_courses, vector_store):
        """Test that exact match is marked in course result."""
        search = HybridSearch(vector_store)
        
        results, _ = search.query("CSCI 0320", top_k=5)
        
        # Check for _exact_match field
        for course in results:
            assert "_exact_match" in course
            if course["course_code"] == "CSCI0320":
                assert course["_exact_match"] is True


class TestSearchIntegration:
    """Integration tests for full retrieval pipeline with structured metadata."""
    
    def test_end_to_end_exact_code_query(self, sample_courses, vector_store):
        """Test exact course code query returns full metadata."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("Who teaches ENGN0030, and when does the class meet?", top_k=5)
        
        assert detected_code == "ENGN0030"
        assert len(results) > 0
        
        # First result should be exact match with all metadata
        first = results[0]
        assert first["course_code"] == "ENGN0030"
        assert first.get("instructor") == "D. Mittleman"
        assert first.get("meeting_times") == "MWF 9:00-9:50"
        assert first["_exact_match"] is True
    
    def test_end_to_end_semantic_query(self, sample_courses, vector_store):
        """Test semantic query without exact code match."""
        search = HybridSearch(vector_store)
        
        results, detected_code = search.query("machine learning courses on Fridays", top_k=5)
        
        assert detected_code is None
        assert len(results) > 0
        assert all("course_code" in r for r in results)
        assert all("_hybrid_score" in r for r in results)
    
    def test_missing_instructor_not_empty(self, sample_courses, vector_store):
        """Test that missing instructor field doesn't break retrieval."""
        search = HybridSearch(vector_store)
        
        results, _ = search.query("MATH0520", top_k=5)
        assert len(results) > 0
        
        # Find MATH course
        math_course = next((c for c in results if c["course_code"] == "MATH0520"), None)
        assert math_course is not None
        # Should have empty string or valid value, not KeyError
        assert "instructor" in math_course


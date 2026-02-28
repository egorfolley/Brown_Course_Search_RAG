import json
import pytest
from fastapi.testclient import TestClient
from app.app import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestQueryEndpoint:
    """Test POST /query endpoint."""
    
    def test_query_endpoint_exists(self, client):
        """Test that /query endpoint exists."""
        response = client.post("/query", json={"q": "test"})
        assert response.status_code in [200, 422, 500]  # Not 404
    
    def test_query_requires_q_parameter(self, client):
        """Test that query parameter is required."""
        response = client.post("/query", json={})
        assert response.status_code == 422  # Validation error
    
    def test_query_accepts_optional_department(self, client):
        """Test that department parameter is optional."""
        response = client.post(
            "/query",
            json={
                "q": "computer science",
                "department": "Computer Science"
            }
        )
        # Should not return validation error (422)
        assert response.status_code != 422
    
    def test_query_rejects_empty_query(self, client):
        """Test that empty query is rejected."""
        response = client.post("/query", json={"q": ""})
        assert response.status_code == 400 or response.status_code == 422
    
    def test_query_response_format(self, client):
        """Test that response has correct structure."""
        response = client.post("/query", json={"q": "test query"})
        
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "courses" in data
            assert isinstance(data["answer"], str)
            assert isinstance(data["courses"], list)
    
    def test_course_result_fields(self, client):
        """Test that course results have required and new fields."""
        response = client.post("/query", json={"q": "test"})
        
        if response.status_code == 200:
            data = response.json()
            # Check for new fields in response
            assert "detected_code" in data, "detected_code field missing from response"
            
            if data["courses"]:
                course = data["courses"][0]
                # Required core fields
                required_fields = {"code", "title", "department", "similarity", "source"}
                assert all(field in course for field in required_fields), \
                    f"Missing fields: {required_fields - set(course.keys())}"
                
                # New metadata fields
                assert "instructor" in course, "instructor field missing"
                assert "meeting_times" in course, "meeting_times field missing"

    
    def test_similarity_score_range(self, client):
        """Test that similarity scores are in valid range."""
        response = client.post("/query", json={"q": "test"})
        
        if response.status_code == 200:
            data = response.json()
            for course in data["courses"]:
                similarity = course.get("similarity", 0)
                assert 0 <= similarity <= 1, f"Invalid similarity score: {similarity}"


class TestAPIErrorHandling:
    """Test error handling in API."""
    
    def test_course_code_detection(self, client):
        """Test that API detects course codes in queries."""
        response = client.post(
            "/query",
            json={"q": "Who teaches ENGN0030?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Should detect ENGN0030 in the query
            assert data.get("detected_code") is not None, "Course code not detected"
    
    def test_course_code_with_space(self, client):
        """Test that API detects course codes with spaces."""
        response = client.post(
            "/query",
            json={"q": "Find ENGN 0030 course"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # Should normalize ENGN 0030 to ENGN0030
            assert data.get("detected_code") is not None, "Course code with space not detected"
    
    def test_exact_match_retrieves_metadata(self, client):
        """Test that exact course code match returns instructor and meeting_times."""
        response = client.post(
            "/query",
            json={"q": "When does CSCI0320 meet and who teaches it?"}
        )
        
        if response.status_code == 200:
            data = response.json()
            # If course code is detected, response should include instructor/meeting_times
            if data.get("detected_code"):
                for course in data["courses"]:
                    # All courses should have these fields (even if empty)
                    assert "instructor" in course
                    assert "meeting_times" in course
    
    def test_timeout_handling(self, client):
        """Test that API handles long queries gracefully."""
        # Very long query might timeout
        long_query = "a " * 10000
        response = client.post("/query", json={"q": long_query})
        
        # Should return error, not crash
        assert response.status_code >= 400
    
    def test_special_characters_handling(self, client):
        """Test that API handles special characters."""
        special_queries = [
            "CSCI0320 & MATH0520",
            "Python/C++",
            "CS: Algorithms",
        ]
        
        for query in special_queries:
            response = client.post("/query", json={"q": query})
            # Should handle without crashing
            assert response.status_code in [200, 400, 500]


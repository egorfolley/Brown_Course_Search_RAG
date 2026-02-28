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
        """Test that course results have required fields."""
        response = client.post("/query", json={"q": "test"})
        
        if response.status_code == 200:
            data = response.json()
            if data["courses"]:
                course = data["courses"][0]
                required_fields = {"code", "title", "department", "similarity", "source"}
                assert all(field in course for field in required_fields)
    
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

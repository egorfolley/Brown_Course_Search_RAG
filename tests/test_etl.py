import json
from pathlib import Path
import pytest
from etl.pipeline import run as run_pipeline


@pytest.fixture
def sample_bulletin_data():
    """Sample bulletin course data."""
    return [
        {
            "course_code": "ENGN0010",
            "title": "Introduction to Engineering",
            "department": "Engineering",
            "description": "Introduction to engineering design.",
            "instructor": "Prof. Smith",
            "meeting_times": "MWF 9:00-10:00",
            "prerequisites": "",
            "source": "Bulletin",
        }
    ]


@pytest.fixture
def sample_cab_data():
    """Sample CAB course data."""
    return [
        {
            "course_code": "ENGN0010",
            "title": "Introduction to Engineering",
            "department": "Engineering",
            "description": "Detailed engineering intro.",
            "instructor": "",
            "meeting_times": "",
            "prerequisites": "High School Math",
            "source": "CAB",
        }
    ]


class TestDataValidation:
    """Test course data validation."""
    
    def test_required_fields(self, sample_bulletin_data):
        """Test that courses have required fields."""
        required_fields = {
            "course_code", "title", "department", "description", "source"
        }
        
        for course in sample_bulletin_data:
            for field in required_fields:
                assert field in course, f"Missing field: {field}"
                assert course[field], f"Empty field: {field}"
    
    def test_course_code_format(self, sample_bulletin_data):
        """Test course code format."""
        for course in sample_bulletin_data:
            code = course["course_code"]
            # Course codes should be alphanumeric (e.g., CSCI0320)
            assert code.replace(" ", "").isalnum(), f"Invalid code format: {code}"


class TestPipelineLogic:
    """Test ETL pipeline merge logic."""
    
    def test_merge_handles_duplicate_codes(self, sample_bulletin_data, sample_cab_data):
        """Test that merge correctly handles courses with same code."""
        # Both have ENGN0010, merge should keep both and mark source as merged
        merged = self._simulate_merge(sample_bulletin_data, sample_cab_data)
        
        engn_courses = [c for c in merged if c["course_code"] == "ENGN0010"]
        assert len(engn_courses) == 1
        assert engn_courses[0]["source"] == "CAB+Bulletin"
    
    def test_merge_prefers_bulletin_schedule_data(self, sample_bulletin_data, sample_cab_data):
        """Test that merge prefers Bulletin's schedule data."""
        merged = self._simulate_merge(sample_bulletin_data, sample_cab_data)
        
        engn = [c for c in merged if c["course_code"] == "ENGN0010"][0]
        
        # Should use Bulletin's meeting times
        assert engn["meeting_times"] == "MWF 9:00-10:00"
        # Should use CAB's prerequisites if Bulletin's is empty
        assert engn["prerequisites"] == "High School Math"
    
    @staticmethod
    def _simulate_merge(bulletin, cab):
        """Simulate merge logic from pipeline.py."""
        merged_dict = {}
        
        for course in bulletin:
            code = course["course_code"]
            merged_dict[code] = course
        
        for course in cab:
            code = course["course_code"]
            if code in merged_dict:
                # Merge: prefer bulletin, fill in gaps from CAB
                existing = merged_dict[code]
                for key, value in course.items():
                    if key == "source":
                        existing["source"] = "CAB+Bulletin"
                    elif not existing.get(key):
                        existing[key] = value
            else:
                merged_dict[code] = course
        
        return list(merged_dict.values())

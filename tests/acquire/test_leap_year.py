"""Test for leap year handling in time window acquisition."""

import pytest
from datetime import datetime

from blackmarble.acquire.landsat_search import search_best_scenes


class TestLeapYear:
    """Test leap year edge cases."""
    
    @pytest.mark.asyncio
    async def test_leap_year_date_iteration(self):
        """Test that date iteration works correctly across leap year boundaries."""
        # Feb 29, 2024 is a leap year
        leap_date = datetime(2024, 2, 29)
        
        # Mock find_scenes function to return empty list
        async def mock_find_scenes(*args, **kwargs):
            return []
        
        # This should not raise ValueError about invalid date
        scenes = await search_best_scenes(
            bbox=(-73.0, 40.5, -72.5, 41.0),
            target_date=leap_date,
            allow_incomplete=True,  # Allow empty results
            find_scenes_func=mock_find_scenes,
        )
        
        # Verify it didn't crash
        assert isinstance(scenes, list)
    
    @pytest.mark.asyncio 
    async def test_year_spanning_search_window(self):
        """Test search window that spans multiple years including leap year."""
        # Search window from Dec 2023 to Feb 2024 (crosses leap year)
        target_date = datetime(2024, 1, 15)
        
        # Track which years were searched
        searched_years = []
        
        async def mock_find_scenes(*args, **kwargs):
            searched_years.append(kwargs.get('year'))
            return []
        
        scenes = await search_best_scenes(
            bbox=(-73.0, 40.5, -72.5, 41.0),
            target_date=target_date,
            allow_incomplete=True,
            find_scenes_func=mock_find_scenes,
        )
        
        # Should have searched both 2023 and 2024
        assert 2023 in searched_years
        assert 2024 in searched_years
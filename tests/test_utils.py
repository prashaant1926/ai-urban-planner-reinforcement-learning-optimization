"""Tests for utility functions."""

import pytest
import tempfile
import os
import json

# Import will work when utils.py is in the path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import chunk_list, flatten_dict, load_json, save_json


class TestChunkList:
    """Tests for chunk_list function."""

    def test_even_chunks(self):
        result = chunk_list([1, 2, 3, 4, 5, 6], 2)
        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_uneven_chunks(self):
        result = chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]

    def test_single_chunk(self):
        result = chunk_list([1, 2, 3], 10)
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = chunk_list([], 5)
        assert result == []


class TestFlattenDict:
    """Tests for flatten_dict function."""

    def test_simple_dict(self):
        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"a": {"b": {"c": 1}}}
        assert flatten_dict(d) == {"a.b.c": 1}

    def test_mixed_dict(self):
        d = {"a": 1, "b": {"c": 2, "d": 3}}
        result = flatten_dict(d)
        assert result == {"a": 1, "b.c": 2, "b.d": 3}

    def test_custom_separator(self):
        d = {"a": {"b": 1}}
        assert flatten_dict(d, sep="/") == {"a/b": 1}


class TestJsonIO:
    """Tests for JSON load/save functions."""

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            data = {"name": "test", "values": [1, 2, 3]}

            save_json(data, filepath)
            loaded = load_json(filepath)

            assert loaded == data

    def test_nested_directory_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "a", "b", "c", "test.json")
            data = {"key": "value"}

            save_json(data, filepath)
            assert os.path.exists(filepath)

            loaded = load_json(filepath)
            assert loaded == data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

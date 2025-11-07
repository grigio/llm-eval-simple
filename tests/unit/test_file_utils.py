"""Unit tests for file utilities module."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from file_utils import (
    get_prompt_files,
    read_file_content,
    write_json_report,
    read_json_report,
    ensure_directory_exists,
    save_raw_results,
    load_raw_results,
    save_evaluated_results,
    load_evaluated_results
)


class TestGetPromptFiles:
    """Test get_prompt_files function."""
    
    def test_get_prompt_files_valid(self, temp_dir):
        """Test getting valid prompt files."""
        # Create test files
        (temp_dir / "test1.txt").write_text("content1")
        (temp_dir / "test2.md").write_text("content2")
        (temp_dir / "test3.py").write_text("content3")
        (temp_dir / "not_allowed.exe").write_text("content4")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "test4.txt").write_text("content4")
        
        # Test pattern matching
        pattern = str(temp_dir / "*.txt")
        files = get_prompt_files(pattern)
        
        assert len(files) == 1
        assert str(temp_dir / "test1.txt") in files
    
    def test_get_prompt_files_no_files(self, temp_dir):
        """Test getting files when none exist."""
        pattern = str(temp_dir / "*.txt")
        files = get_prompt_files(pattern)
        
        assert files == []
    
    def test_get_prompt_files_invalid_pattern(self):
        """Test invalid pattern handling."""
        with patch('validation.validate_glob_pattern') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid pattern")
            
            files = get_prompt_files("invalid/*")
            
            assert files == []
    
    def test_get_prompt_files_oversized_file(self, temp_dir):
        """Test handling of oversized files."""
        # Create oversized file
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * (11 * 1024 * 1024))  # 11MB
        
        pattern = str(temp_dir / "*.txt")
        files = get_prompt_files(pattern)
        
        # Oversized files should be skipped
        assert large_file not in [Path(f) for f in files]
    
    def test_get_prompt_files_sorted(self, temp_dir):
        """Test that files are returned sorted."""
        (temp_dir / "z_file.txt").write_text("content")
        (temp_dir / "a_file.txt").write_text("content")
        (temp_dir / "m_file.txt").write_text("content")
        
        pattern = str(temp_dir / "*.txt")
        files = get_prompt_files(pattern)
        
        # Check if files are sorted
        file_names = [Path(f).name for f in files]
        assert file_names == sorted(file_names)


class TestReadFileContent:
    """Test read_file_content function."""
    
    def test_read_file_content_valid(self, sample_prompt_file):
        """Test reading valid file content."""
        content = read_file_content(str(sample_prompt_file))
        
        assert content == "What is 2 + 2?"
    
    def test_read_file_content_with_whitespace(self, temp_dir):
        """Test reading file with extra whitespace."""
        test_file = temp_dir / "whitespace.txt"
        test_file.write_text("  content with spaces  \n\n")
        
        content = read_file_content(str(test_file))
        
        assert content == "content with spaces"
    
    def test_read_file_content_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(ValueError, match="Error reading file"):
            read_file_content("non_existent_file.txt")
    
    def test_read_file_content_invalid_encoding(self, temp_dir):
        """Test reading file with invalid encoding."""
        invalid_file = temp_dir / "invalid.txt"
        invalid_file.write_bytes(b'\xff\xfe\x00\x00')  # Invalid UTF-8
        
        with pytest.raises(ValueError, match="is not valid UTF-8 text"):
            read_file_content(str(invalid_file))
    
    def test_read_file_content_oversized(self, temp_dir):
        """Test reading oversized file."""
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * 60000)  # Exceeds 50KB limit
        
        with pytest.raises(ValueError, match="Content too long"):
            read_file_content(str(large_file))
    
    def test_read_file_content_path_traversal(self):
        """Test path traversal protection."""
        with pytest.raises(ValueError, match="Error reading file"):
            read_file_content("../../../etc/passwd")


class TestWriteJsonReport:
    """Test write_json_report function."""
    
    def test_write_json_report_valid(self, temp_dir, sample_results):
        """Test writing valid JSON report."""
        report_file = temp_dir / "report.json"
        
        write_json_report(sample_results, str(report_file))
        
        assert report_file.exists()
        
        # Verify content
        with open(report_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert data == sample_results
    
    def test_write_json_report_creates_directory(self, temp_dir, sample_results):
        """Test that write creates parent directories."""
        nested_dir = temp_dir / "nested" / "dir"
        report_file = nested_dir / "report.json"
        
        write_json_report(sample_results, str(report_file))
        
        assert nested_dir.exists()
        assert report_file.exists()
    
    def test_write_json_report_oversized_data(self, temp_dir):
        """Test writing oversized data."""
        # Create data larger than 100MB
        large_data = [{"x": "y" * 10000} for _ in range(100000)]
        
        report_file = temp_dir / "large.json"
        
        with pytest.raises(ValueError, match="Data too large to write"):
            write_json_report(large_data, str(report_file))
    
    def test_write_json_report_invalid_path(self):
        """Test writing to invalid path."""
        with pytest.raises(ValueError, match="Error writing file"):
            write_json_report([], "../../../etc/report.json")
    
    def test_write_json_report_atomic_write(self, temp_dir, sample_results):
        """Test atomic write behavior."""
        report_file = temp_dir / "report.json"
        
        write_json_report(sample_results, str(report_file))
        
        # Verify no temporary file left behind
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0


class TestReadJsonReport:
    """Test read_json_report function."""
    
    def test_read_json_report_valid(self, sample_json_file):
        """Test reading valid JSON report."""
        data = read_json_report(str(sample_json_file))
        
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["model"] == "test-model"
    
    def test_read_json_report_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(ValueError, match="Error reading file"):
            read_json_report("non_existent.json")
    
    def test_read_json_report_invalid_json(self, invalid_json_file):
        """Test reading invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            read_json_report(str(invalid_json_file))
    
    def test_read_json_report_not_a_list(self, temp_dir):
        """Test reading JSON that's not a list."""
        not_a_list_file = temp_dir / "not_list.json"
        not_a_list_file.write_text('{"key": "value"}')
        
        with pytest.raises(ValueError, match="JSON data must be a list"):
            read_json_report(str(not_a_list_file))
    
    def test_read_json_report_invalid_items(self, temp_dir):
        """Test reading JSON with invalid items."""
        invalid_items_file = temp_dir / "invalid_items.json"
        invalid_items_file.write_text('[{"valid": "dict"}, "not a dict"]')
        
        with pytest.raises(ValueError, match="Item 1 must be a dictionary"):
            read_json_report(str(invalid_items_file))
    
    def test_read_json_report_oversized(self, temp_dir):
        """Test reading oversized JSON file."""
        oversized_file = temp_dir / "oversized.json"
        oversized_file.write_text("x" * (11 * 1024 * 1024))  # 11MB
        
        with pytest.raises(ValueError, match="File too large"):
            read_json_report(str(oversized_file))


class TestEnsureDirectoryExists:
    """Test ensure_directory_exists function."""
    
    def test_ensure_directory_exists_new(self, temp_dir):
        """Test creating new directory."""
        new_dir = temp_dir / "new_directory"
        
        ensure_directory_exists(str(new_dir))
        
        assert new_dir.exists()
        assert new_dir.is_dir()
    
    def test_ensure_directory_exists_existing(self, temp_dir):
        """Test with existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        # Should not raise an error
        ensure_directory_exists(str(existing_dir))
        
        assert existing_dir.exists()
    
    def test_ensure_directory_exists_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        ensure_directory_exists(str(nested_dir))
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_ensure_directory_exists_invalid_path(self):
        """Test invalid directory path."""
        with pytest.raises(ValueError, match="Error creating directory"):
            ensure_directory_exists("../../../etc/invalid")


class TestSaveLoadRawResults:
    """Test save_raw_results and load_raw_results functions."""
    
    def test_save_and_load_raw_results(self, sample_results):
        """Test saving and loading raw results."""
        save_raw_results(sample_results)
        
        loaded_results = load_raw_results()
        
        assert loaded_results == sample_results
    
    @patch('file_utils.write_json_report')
    def test_save_raw_results_calls_write(self, mock_write, sample_results):
        """Test that save_raw_results calls write_json_report."""
        save_raw_results(sample_results)
        
        mock_write.assert_called_once_with(sample_results, 'answers-generated/report.json')
    
    @patch('file_utils.read_json_report')
    def test_load_raw_results_calls_read(self, mock_read, sample_results):
        """Test that load_raw_results calls read_json_report."""
        mock_read.return_value = sample_results
        
        result = load_raw_results()
        
        mock_read.assert_called_once()
        assert result == sample_results


class TestSaveLoadEvaluatedResults:
    """Test save_evaluated_results and load_evaluated_results functions."""
    
    def test_save_and_load_evaluated_results(self, sample_results):
        """Test saving and loading evaluated results."""
        save_evaluated_results(sample_results)
        
        loaded_results = load_evaluated_results()
        
        assert loaded_results == sample_results
    
    @patch('file_utils.write_json_report')
    def test_save_evaluated_results_calls_write(self, mock_write, sample_results):
        """Test that save_evaluated_results calls write_json_report."""
        save_evaluated_results(sample_results)
        
        mock_write.assert_called_once_with(sample_results, 'answers-generated/report-evaluated.json')
    
    @patch('file_utils.read_json_report')
    def test_load_evaluated_results_calls_read(self, mock_read, sample_results):
        """Test that load_evaluated_results calls read_json_report."""
        mock_read.return_value = sample_results
        
        result = load_evaluated_results()
        
        mock_read.assert_called_once()
        assert result == sample_results


@pytest.mark.unit
class TestFileUtilitiesIntegration:
    """Integration tests for file utilities."""
    
    def test_complete_file_workflow(self, temp_dir, sample_results):
        """Test complete file workflow."""
        # Create directory
        reports_dir = temp_dir / "reports"
        ensure_directory_exists(str(reports_dir))
        
        # Write report
        report_file = reports_dir / "test_report.json"
        write_json_report(sample_results, str(report_file))
        
        # Read report
        loaded_data = read_json_report(str(report_file))
        
        assert loaded_data == sample_results
        
        # Verify file exists and has correct content
        assert report_file.exists()
        
        # Verify JSON is properly formatted
        with open(report_file, 'r') as f:
            content = f.read()
            # Should be properly indented
            assert '\n  ' in content
    
    def test_error_handling_workflow(self, temp_dir):
        """Test error handling throughout workflow."""
        # Try to read non-existent file
        with pytest.raises(ValueError):
            read_file_content(str(temp_dir / "non_existent.txt"))
        
        # Try to write to invalid location
        with pytest.raises(ValueError):
            write_json_report([], "/invalid/path/report.json")
        
        # Try to read invalid JSON
        invalid_file = temp_dir / "invalid.json"
        invalid_file.write_text("{ invalid json")
        
        with pytest.raises(ValueError):
            read_json_report(str(invalid_file))
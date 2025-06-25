import sys
import os
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from unittest import mock
import train as train_module

def test_train_model_runs(monkeypatch, capsys):
    # Mock MLflow functions to avoid actual logging
    mock_start_run = mock.MagicMock()
    mock_log_param = mock.MagicMock()
    mock_log_metric = mock.MagicMock()
    mock_log_model = mock.MagicMock()
    
    # Mock mlflow.start_run() context manager
    class MockRun:
        info = mock.MagicMock(run_id="12345")
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): return False
    
    monkeypatch.setattr(train_module.mlflow, "start_run", lambda: MockRun())
    monkeypatch.setattr(train_module.mlflow, "log_param", mock_log_param)
    monkeypatch.setattr(train_module.mlflow, "log_metric", mock_log_metric)
    monkeypatch.setattr(train_module.mlflow.sklearn, "log_model", mock_log_model)
    
    # Run the training function
    with tempfile.TemporaryDirectory() as tmpdir:
        train_module.train_model(model_output_path=tmpdir)
    
    # Capture printed output
    captured = capsys.readouterr()
    assert "Starting training..." in captured.out
    assert "Model trained" in captured.out
    assert "Model accuracy" in captured.out
    assert "Run ID" in captured.out
    
    # Assert MLflow logging functions called
    mock_log_param.assert_called_once_with("max_iter", 200)
    mock_log_metric.assert_called_once()
    mock_log_model.assert_called_once()

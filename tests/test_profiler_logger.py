import time
import numpy as np
from core.profiler import StepProfiler
from core.data_logger import DataLogger

def test_profiler_measure():
    profiler = StepProfiler()
    for _ in range(100):
        with profiler.measure("test"):
            time.sleep(0.0001)
    
    summary = profiler.summary()
    assert summary["test"]["count"] == 100
    assert summary["test"]["mean_us"] > 0

def test_profiler_disabled():
    profiler = StepProfiler()
    profiler.enabled = False
    with profiler.measure("test"):
        time.sleep(0.0001)
    
    assert "test" not in profiler.summary()

def test_profiler_reset():
    profiler = StepProfiler()
    with profiler.measure("test"):
        time.sleep(0.0001)
    
    profiler.reset()
    assert len(profiler.summary()) == 0

def test_logger_log_get():
    logger = DataLogger()
    logger.log("ch", 42, t=1.0)
    data = logger.get("ch")
    assert data == [(1.0, 42)]

def test_logger_channels():
    logger = DataLogger()
    logger.log("ch1", 1, 0.1)
    logger.log("ch2", 2, 0.2)
    logger.log("ch3", 3, 0.3)
    channels = logger.channels()
    assert set(channels) == {"ch1", "ch2", "ch3"}

def test_logger_log_state():
    logger = DataLogger()
    state = np.array([1, 2, 3])
    logger.log_state(state, t=0.5)
    data = logger.get("state")
    assert isinstance(data[0][1], list)
    assert data[0][1] == [1, 2, 3]

def test_logger_export_npz(tmp_path):
    logger = DataLogger()
    for i in range(10):
        logger.log_state(np.array([i, i*2]), t=i*0.1)
    
    filepath = tmp_path / "test.npz"
    logger.export_npz(filepath)
    
    loaded = np.load(filepath)
    assert "state_t" in loaded
    assert "state_v" in loaded
    assert loaded["state_v"].shape == (10, 2)

def test_logger_export_csv(tmp_path):
    logger = DataLogger()
    for i in range(10):
        logger.log_cost(float(i), t=i*0.1)
    
    filepath = tmp_path / "test.csv"
    logger.export_csv(filepath, channel="cost")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 11  # header + 10 data rows

def test_logger_max_entries():
    logger = DataLogger(max_entries=5)
    for i in range(10):
        logger.log("ch", i, t=i*0.1)
    
    assert len(logger.get("ch")) == 5

def test_logger_clear():
    logger = DataLogger()
    logger.log("ch", 1, 0.1)
    logger.clear()
    assert len(logger.channels()) == 0

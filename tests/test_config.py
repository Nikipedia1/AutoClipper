"""Basic tests for config module."""
import config


def test_directories_exist():
    config.ensure_directories()
    assert config.INPUT_DIR.exists()
    assert config.TEMP_DIR.exists()
    assert config.OUTPUT_DIR.exists()
    assert config.PROCESSED_LOGS_DIR.exists()


def test_target_resolution():
    assert config.TARGET_WIDTH == 1080
    assert config.TARGET_HEIGHT == 1920


def test_clip_duration_limits():
    assert config.MIN_CLIP_SEC < config.MAX_CLIP_SEC
    assert config.MIN_CLIP_SEC >= 15
    assert config.MAX_CLIP_SEC <= 90


def test_smart_crop_default():
    assert isinstance(config.SMART_CROP, bool)


def test_device_is_valid():
    assert config.DEVICE in ("cpu", "cuda")


def test_compute_type_is_valid():
    assert config.COMPUTE_TYPE in ("int8", "float16", "float32", "int8_float16")

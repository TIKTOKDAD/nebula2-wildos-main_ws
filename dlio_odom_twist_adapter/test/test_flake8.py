"""运行 DLIO twist 转换包的 ROS 2 Python 风格检查."""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """检查节点、数学模块、launch 和测试代码的 flake8 规则."""
    return_code, errors = main_with_errors(
        argv=["--config", "test/ament_flake8.ini"]
    )
    assert return_code == 0, (
        "Found %d code style errors / warnings:\n" % len(errors)
        + "\n".join(errors)
    )

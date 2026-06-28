"""运行 graphnav_builder 的 ROS 2 Python 风格检查。"""

from ament_flake8.main import main_with_errors
import pytest


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    """检查图构建、启动和测试源码的 flake8 格式规则。"""
    rc, errors = main_with_errors(argv=[])
    assert rc == 0, (
        'Found %d code style errors / warnings:\n' % len(errors)
        + '\n'.join(errors)
    )

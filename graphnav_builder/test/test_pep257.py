"""运行 graphnav_builder 的 ROS 2 文档字符串检查。"""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """检查公开建图 API、启动文件和测试辅助代码的文档字符串。"""
    rc = main(argv=['.', 'test'])
    assert rc == 0, 'Found code style errors / warnings'

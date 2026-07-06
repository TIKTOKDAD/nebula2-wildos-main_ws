"""运行 DLIO twist 转换包的 Python 文档字符串检查."""

from ament_pep257.main import main
import pytest


@pytest.mark.linter
@pytest.mark.pep257
def test_pep257():
    """检查运行代码和 launch 文件的公开 API 文档字符串."""
    return_code = main(argv=["dlio_odom_twist_adapter", "launch"])
    assert return_code == 0, "Found code style errors / warnings"

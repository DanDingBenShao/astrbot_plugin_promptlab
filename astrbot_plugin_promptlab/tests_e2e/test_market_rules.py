"""市场规范守卫测试：AstrBot 市场审查红线，一条都不能碰。"""
from __future__ import annotations

import os
import re
import sys

PLUGIN_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

IMPORT_BAN_RE = re.compile(
    r"^(?:import\s+(?:logging|loguru)\b|from\s+(?:logging|loguru)\b)")


def _plugin_files():
    for root, _dirs, files in os.walk(PLUGIN_DIR):
        if "tests_e2e" in root or "__pycache__" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                yield os.path.join(root, fn)


def test_no_logging_or_loguru_import():
    """日志只能 from astrbot.api import logger。"""
    for path in _plugin_files():
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                assert not IMPORT_BAN_RE.match(line.strip()), \
                    f"{path}:{i} 违规 import：{line.strip()}"


def test_no_sqlite3_direct():
    """禁 sqlite3 直连官方 data_v4.db。"""
    for path in _plugin_files():
        with open(path, encoding="utf-8") as f:
            src = f.read()
        assert "import sqlite3" not in src, f"{path} 出现 sqlite3"


def test_no_contexts_replacement():
    """不注册 on_llm_request/on_llm_response 钩子、不赋值 req.contexts——
    从机制上不存在上下文整体替换的可能（按代码语法匹配，docstring 提及不算）。"""
    ban_re = re.compile(
        r"@(?:filter\.)?on_llm_(?:request|response)|req\.contexts\s*=")
    for path in _plugin_files():
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                assert not ban_re.search(line), \
                    f"{path}:{i} 违规：{line.strip()}"


def test_data_in_plugin_data_dir():
    """数据必须落 data/plugin_data/astrbot_plugin_promptlab/。"""
    from astrbot_plugin_promptlab import main
    root = main._plugin_data_root()
    assert "plugin_data" in root and "astrbot_plugin_promptlab" in root


def test_official_conversation_api_only():
    """会话历史只允许走 get_db().get_conversations()。"""
    from astrbot_plugin_promptlab import main
    assert hasattr(main.PromptLabPlugin, "_collect_samples")
    import inspect
    src = inspect.getsource(main.PromptLabPlugin._collect_samples)
    assert "get_conversations" in src
    assert "data_v4" not in src

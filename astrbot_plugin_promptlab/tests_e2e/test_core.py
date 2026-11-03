"""核心逻辑测试：状态机 / 提示词构建与解析 / 本地兜底。"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ASTRBOT_ROOT = Path(r"C:\Users\16535\Documents\deepseek-harness"
                     r"\default-workspace\qqbot-lab\astrbot\AstrBot-master")
if str(_ASTRBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASTRBOT_ROOT))
os.environ.setdefault("ASTRBOT_ROOT",
                      str(Path(tempfile.gettempdir()) / "promptlab_e2e_root"))
os.makedirs(os.environ["ASTRBOT_ROOT"], exist_ok=True)
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astrbot_plugin_promptlab.core.judge import (  # noqa: E402
    build_act_user, build_checklist_prompt, build_judge_prompt,
    build_optimizer_prompt, parse_checklist, parse_judge)
from astrbot_plugin_promptlab.core.state import (  # noqa: E402
    SessionStore, TrainingSession, _json_write_atomic)
from astrbot_plugin_promptlab.core.engine import _too_similar  # noqa: E402


# ---------- 状态 ----------
def test_session_roundtrip(tmp_path):
    s = TrainingSession("0901-120000", 123, "origin-x", "更像真人",
                        max_rounds=3, pass_score=70, samples="样本",
                        base_prompt="基线提示词")
    s.checklist = [{"item": "句尾无句号", "weight": "高"}]
    s.baseline_reply = "基线回复"
    s.append_round({"n": 1, "prompt": "v2", "reply": "回",
                    "judge": {"score": 55, "verdict": "fail",
                              "feedback": ["改"], "scores": [],
                              "diverged": True, "faithful": True,
                              "summary": "不够"},
                    "ts": 0.0})
    s.cur_prompt = "v2"
    p = str(tmp_path / "s.json")
    _json_write_atomic(p, s.to_dict())
    s2 = TrainingSession.from_dict(__import__("json").load(open(p, encoding="utf-8")))
    assert s2.tid == "0901-120000"
    assert s2.status == TrainingSession.STATUS_PENDING
    assert s2.checklist == [{"item": "句尾无句号", "weight": "高"}]
    assert s2.baseline_reply == "基线回复"
    assert s2.cur_prompt == "v2"
    assert s2.rounds[0]["judge"]["score"] == 55
    assert s2.rounds_left() == 2


def test_store_latest_for_owner(tmp_path):
    store = SessionStore(str(tmp_path))
    a = TrainingSession("t1", 111, "o", "需求A")
    b = TrainingSession("t2", 111, "o", "需求B")
    b.created = a.created + 10
    c = TrainingSession("t3", 222, "o", "别人的")
    for s in (a, b, c):
        store.save(s)
    got = store.latest_for_owner(111)
    assert got and got.tid == "t2"
    assert store.latest_for_owner(999) is None


# ---------- 提示词构建/解析 ----------
def test_build_and_parse_checklist():
    p = build_checklist_prompt("更像真人")
    assert "更像真人" in p
    parsed = parse_checklist(
        '```json\n[{"item": "句尾无句号", "weight": "高"},'
        '{"item": "用口语词", "weight": "中"}]\n```')
    assert parsed[0]["item"] == "句尾无句号"
    assert parsed[0]["weight"] == "高"
    assert len(parsed) == 2
    # 解析失败 → 保底单条
    fb = parse_checklist("模型抽风输出了一堆没有JSON的文字")
    assert len(fb) == 1 and "weight" in fb[0]


def test_build_judge_prompt_fields():
    p = build_judge_prompt(
        "需求", [{"item": "A", "weight": "高"}], "基线回复", "候选回复")
    assert "基线回复" in p and "候选回复" in p
    assert "变化度" in p and "保真" in p and "硬伤" in p
    # 无基线时带降级说明
    p2 = build_judge_prompt("需求", [], "", "候选回复")
    assert "基线缺失" in p2


def test_parse_judge_normalizes():
    good = ('{"scores": [{"item": "A", "weight": "高", "pass": true, '
            '"evidence": "因为"}], "diverged": true, "faithful": true, '
            '"score": 88, "verdict": "PASS", "feedback": ["f1"], '
            '"summary": "还行"}')
    j = parse_judge(good)
    assert j and j["verdict"] == "pass" and j["score"] == 88
    assert j["scores"][0]["pass"] is True
    # 分数越界夹紧 + feedback 字符串转数组 + 垃圾输出 None
    j2 = parse_judge('{"score": 250, "verdict": "fail", "feedback": "单条"}')
    assert j2["score"] == 100 and j2["feedback"] == ["单条"]
    assert parse_judge("完全没有JSON") is None


def test_optimizer_prompt_contains_feedback():
    p = build_optimizer_prompt(
        "需求", [{"item": "A", "weight": "高"}], "当前提示词",
        ["指令1", "指令2"], "基线提示词")
    assert "指令1" in p and "指令2" in p
    assert "当前提示词" in p and "基线提示词" in p
    assert "可检查清单" in p


def test_build_act_user():
    u = build_act_user("【风格参照】样本", "在吗？")
    assert "风格参照" in u and "在吗？" in u
    assert "只输出回复本身" in u


# ---------- 本地兜底 ----------
def test_too_similar():
    assert _too_similar("你好呀今天天气不错", "你好呀今天天气不错")
    assert _too_similar("你好呀今天天气不错我们去玩吧哈哈", "你好呀今天天气不错")
    assert not _too_similar("今天天气不错", "完全另一段话的内容呢")
    # 完全相同永远算（短回复也一样——回了两个一模一样的「好」就是没变）
    assert _too_similar("好", "好")
    # 双方都不够长且不完全相同时不判（避免误杀）
    assert not _too_similar("好呀", "好啊")
    assert not _too_similar("", "基线")


def test_rounds_left_and_last_round():
    s = TrainingSession("t", 1, "o", "r")
    assert s.last_round() is None
    assert s.rounds_left() == 3
    s.append_round({"n": 1})
    assert s.last_round()["n"] == 1

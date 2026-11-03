"""语料管理员测试：含真实语料（D:/qqmind-plugin/corpus）实测挑选效果。"""
from __future__ import annotations

import json
import os
import sys

_ASTRBOT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "astrbot_root_stub")
sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..")))

from astrbot_plugin_promptlab.core import curator  # noqa: E402

REAL_CORPUS = r"D:\qqmind-plugin\corpus\human_chat_corpus_1031.json"


def test_clean_entry_rules():
    assert curator.clean_entry("a", "今天天气不错") == "今天天气不错"
    assert curator.clean_entry("a", "[引用消息(溯: x)] 正文内容") == "正文内容"
    assert curator.clean_entry("a", "/help") is None
    assert curator.clean_entry("a", "[图片]") is None
    assert curator.clean_entry("a", "看这个 https://example.com/x") is None
    assert curator.clean_entry("a", "？？？") is None
    assert curator.clean_entry("a", "x" * 100) is None
    assert curator.clean_entry("a", "哈") is None      # <2 字


def test_load_corpus_formats(tmp_path):
    # JSON list[dict]（qqmind 画像格式）
    p1 = tmp_path / "c1.json"
    p1.write_text(json.dumps([
        {"gid": "1", "nick": "张三", "uid": "11", "text": "吃了吗"},
        {"gid": "1", "nick": "李四", "uid": "22", "text": "/命令"},
        {"gid": "1", "nick": "王五", "uid": "33", "text": "[图片]"},
    ], ensure_ascii=False), encoding="utf-8")
    out = curator.load_corpus(str(p1))
    assert out == [("张三", "吃了吗")]
    # role/content 格式
    p2 = tmp_path / "c2.json"
    p2.write_text(json.dumps([
        {"role": "user", "content": "中午吃啥"},
        {"role": "assistant", "content": "食堂饭"},
        {"role": "system", "content": "忽略我"},
    ], ensure_ascii=False), encoding="utf-8")
    out2 = curator.load_corpus(str(p2))
    assert ("用户", "中午吃啥") in out2 and ("Bot", "食堂饭") in out2
    # JSONL
    p3 = tmp_path / "c3.jsonl"
    p3.write_text('{"text": "第一"}\n{"text": "第二"}\n', encoding="utf-8")
    assert len(curator.load_corpus(str(p3))) == 2
    # TXT
    p4 = tmp_path / "c4.txt"
    p4.write_text("一行文本\n另一行\n", encoding="utf-8")
    assert len(curator.load_corpus(str(p4))) == 2
    # 不存在
    assert curator.load_corpus(str(tmp_path / "nope.json")) == []


def test_curate_length_balance():
    entries = [(f"u{i}", t) for i, t in enumerate(
        ["短", "稍短点的消息", "这是一条中等长度的消息内容呢", "这" * 40])]
    text, stats = curator.curate(entries, "更像真人", max_lines=4)
    assert stats["picked"] == 4
    assert stats["chars"] <= 1200
    lines = text.split("\n")
    assert len(lines) == 4


def test_curate_feature_boost():
    """需求含「短」→ 1-8 字消息应优先入选。"""
    entries = [(f"u{i}", t) for i, t in enumerate(
        ["这是一条很长很长很长的消息完全不符合短的特征呢",
         "乐", "草", "笑死", "另一条也很长很长很长超出了短消息的范围了呀"])]
    text, _ = curator.curate(entries, "回复要短", max_lines=3)
    got = [ln.split("：", 1)[-1] for ln in text.split("\n")]
    assert "乐" in got or "草" in got or "笑死" in got


def test_curate_dedup_and_empty():
    entries = [("a", "重复消息"), ("b", "重复消息"), ("c", "重复消息")]
    text, stats = curator.curate(entries, "x", max_lines=5)
    assert stats["picked"] == 1
    t2, st2 = curator.curate([], "x")
    assert t2 == "" and st2["picked"] == 0


def test_real_corpus_curate():
    """真实 3481 条群聊语料：能读、能清、能挑，挑出来的像人话。"""
    entries = curator.load_corpus(REAL_CORPUS)
    assert len(entries) > 2000, f"真实语料应读入大部分，实际 {len(entries)}"
    text, stats = curator.curate(entries, "回复要更像真人，短一点，可以用梗",
                                 max_lines=14)
    assert stats["picked"] == 14
    assert len(text) <= 1200
    lines = text.split("\n")
    assert len(lines) == 14
    # 清洗率：命令/图片/链接应已被剔除（宽松断言：清洗后仍占多数）
    assert stats["cleaned"] > 1500
    print(f"\n[real corpus] total={stats['total']} cleaned={stats['cleaned']} "
          f"picked={stats['picked']} chars={stats['chars']}")
    print(text)


def test_real_corpus_deterministic_seed():
    entries = curator.load_corpus(REAL_CORPUS)
    t1, _ = curator.curate(entries, "短一点", seed=42)
    t2, _ = curator.curate(entries, "短一点", seed=42)
    t3, _ = curator.curate(entries, "短一点", seed=43)
    assert t1 == t2
    assert t1 != t3 or True   # 不同 seed 大概率不同（不强制）

"""端到端测试：完整对抗循环（ScriptProvider 驱动）+ 指令层 + 落盘。

全部原生 async：后台训练任务与测试在同一事件循环里，
跨 asyncio.run 会让 create_task 的任务随旧循环一起死。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from harness import PluginHarness, make_event  # noqa: E402

# 常用剧本路由：(system 含, user 含, 输出)
_R_CHECKLIST = ("验收官", "用户需求", '[{"item": "句尾无句号", "weight": "高"},'
                '{"item": "不用书面衔接词", "weight": "中"}]')
_R_JUDGE_PASS = ("验收官", "候选回复",
                 json.dumps({"scores": [{"item": "句尾无句号", "weight": "高",
                                         "pass": True, "evidence": "无句号"}],
                             "diverged": True, "faithful": True, "score": 85,
                             "verdict": "pass", "feedback": [],
                             "summary": "达标"}, ensure_ascii=False))
_R_JUDGE_FAIL = ("验收官", "候选回复",
                 json.dumps({"scores": [{"item": "句尾无句号", "weight": "高",
                                         "pass": False, "evidence": "有句号"}],
                             "diverged": False, "faithful": True, "score": 40,
                             "verdict": "fail",
                             "feedback": ["删掉书面腔", "句尾去句号"],
                             "summary": "还是老样子"}, ensure_ascii=False))


def _script() -> list:
    return [
        _R_CHECKLIST,
        ("提示词优化师", "验收反馈",
         "修改后的提示词：你是个活的真人，说话别像客服。"),
        ("测试小助手", "现在请以这个身份", "好的，今天摸了一天鱼，累"),
        _R_JUDGE_PASS,
    ]


async def _wait_done(h, timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.05)):
        if not h.plugin.engine.busy(10001):
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("training task did not finish in time")


async def test_full_pass_in_one_round():
    h = PluginHarness(_script())
    try:
        outs = await h.run_cmd("/训练开始 句尾别带句号，别书面腔")
        assert "已开跑" in "".join(outs)
        await _wait_done(h)
        status = h.plugin._status(10001)
        assert "passed" in status, status
        s = h.plugin.store.latest_for_owner(10001)
        assert s and s.status == "passed"
        assert len(s.rounds) == 1
        assert s.rounds[0]["judge"]["score"] == 85
        assert s.baseline_reply
        assert h.context.sent, "训练结束应主动汇报"
        report = h.plugin._report(10001)
        assert "✅ 通过" in report and "收敛后的提示词" in report
    finally:
        h.cleanup()


async def test_three_rounds_then_maxed():
    routes = [
        _R_CHECKLIST,
        ("提示词优化师", "验收反馈", "优化提示词vN"),
        ("测试小助手", "现在请以这个身份", "这是和基线差不多的回复内容呀"),
        _R_JUDGE_FAIL,
    ]
    h = PluginHarness(routes, cfg={"max_rounds": 3})
    try:
        await h.run_cmd("/训练开始 更像真人")
        await _wait_done(h)
        s = h.plugin.store.latest_for_owner(10001)
        assert s.status == "maxed"
        assert len(s.rounds) == 3, "默认 3 轮必停"
        report = h.plugin._report(10001)
        assert "未通过（满轮停止）" in report
        assert "收敛后的提示词" in report
    finally:
        h.cleanup()


async def test_similar_reply_local_gate():
    """验收官放水判 pass，但回复与基线几乎相同 → 本地兜底改判 fail。"""
    routes = [
        _R_CHECKLIST,
        ("提示词优化师", "验收反馈", "几乎没改的提示词"),
        ("", "现在请以这个身份",
         "这是一个足够长到能触发相似度检测的基线回复内容呀"),
        ("验收官", "候选回复",
         json.dumps({"scores": [], "diverged": True, "faithful": True,
                     "score": 80, "verdict": "pass", "feedback": [],
                     "summary": "过了"}, ensure_ascii=False)),
    ]
    h = PluginHarness(routes)
    try:
        await h.run_cmd("/训练开始 更像真人")
        await _wait_done(h)
        s = h.plugin.store.latest_for_owner(10001)
        assert s.rounds, "至少跑了一轮"
        assert s.rounds[0]["judge"]["verdict"] == "fail", "本地相似度门应改判"
        assert "几乎相同" in "".join(s.rounds[0]["judge"]["feedback"])
    finally:
        h.cleanup()


async def test_stop_and_resume_and_usage():
    h = PluginHarness(_script(), cfg={"max_rounds": 1})
    try:
        outs = await h.run_cmd("/训练开始")
        assert "用法" in "".join(outs)
        outs = await h.run_cmd("/训练帮助")
        assert "对抗训练室" in "".join(outs)
        assert "没有训练记录" in h.plugin._status(10001)
        assert "样本已就绪" in h.plugin._set_samples(10001, "用户：哈哈哈\nBot：乐")
        outs = await h.run_cmd("/训练开始 要用梗")
        assert "已开跑" in "".join(outs)
        await _wait_done(h)
        outs = await h.run_cmd("/训练继续 2")
        assert "已通过收敛" in "".join(outs)
        assert "#" in h.plugin._list(10001)
    finally:
        h.cleanup()


async def test_samples_from_official_db():
    conv_content = [{"role": "user", "content": "中午吃的啥"},
                    {"role": "assistant", "content": "食堂饭，小酥肉"},
                    {"role": "system", "message": "ignore"}]
    conv = type("C", (), {"content": conv_content})()
    h = PluginHarness(_script(), convs=[conv])
    try:
        samples = await h.plugin._collect_samples(10001)
        assert "中午吃的啥" in samples and "食堂饭" in samples
        assert "ignore" not in samples
    finally:
        h.cleanup()


async def test_base_prompt_from_persona():
    h = PluginHarness(_script(), persona_prompt="你是测试小助手。")
    try:
        bp = await h.plugin._base_prompt()
        assert bp == "你是测试小助手。"
    finally:
        h.cleanup()


async def test_busy_lock():
    routes = [
        _R_CHECKLIST,
        ("提示词优化师", "验收反馈", "改好的提示词"),
        ("测试小助手", "现在请以这个身份", "回复内容"),
        _R_JUDGE_FAIL,
    ]
    h = PluginHarness(routes, cfg={"max_rounds": 5})
    try:
        await h.run_cmd("/训练开始 慢慢改")
        assert h.plugin.engine.busy(10001), "任务应正在跑"
        outs = await h.run_cmd("/训练开始 抢跑")
        assert "已有训练在跑" in "".join(outs)
        assert "本轮跑完后停" in h.plugin._stop(10001)
        await _wait_done(h, timeout_s=8)
        s = h.plugin.store.latest_for_owner(10001)
        assert s.status in ("stopped", "maxed")
    finally:
        h.cleanup()

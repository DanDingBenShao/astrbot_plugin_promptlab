"""真模型端到端实测：DeepSeek + 真语料 + 真人设，跑一次完整对抗训练。

手动脚本（烧真 API 调用，不进 CI）。运行：
    cd astrbot_plugin_promptlab/tests_e2e && python real_e2e.py

key 从实验室 cmd_config.json 读取（不写入本仓库）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_ASTRBOT_ROOT = Path(r"C:\Users\16535\Documents\deepseek-harness"
                     r"\default-workspace\qqbot-lab\astrbot\AstrBot-master")
for p in (str(_ASTRBOT_ROOT),
          os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ASTRBOT_ROOT",
                      str(Path(tempfile.gettempdir()) / "promptlab_real_root"))
os.makedirs(os.environ["ASTRBOT_ROOT"], exist_ok=True)

import aiohttp  # noqa: E402

# 渠道选择（按顺序探测可用性）：
# 1. 实验室 cmd_config.json 的 DeepSeek 官方 API（可能 402 没余额）
# 2. 本机 wb2a-hub（127.0.0.1:8788，cn 通道，wbhub- key 在 D:/token-relay/.env）
_LAB_CFG = Path(r"C:\Users\16535\Documents\deepseek-harness"
                r"\default-workspace\qqbot-lab\astrbot\AstrBot-master"
                r"\data\cmd_config.json")
_RELAY_ENV = Path(r"D:\token-relay\.env")


def _load_wb2a() -> dict:
    env = {}
    for ln in _RELAY_ENV.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip()
        if "=" in ln and not ln.startswith("#"):
            k, _, v = ln.partition("=")
            env[k.strip()] = v.strip().strip("\"'")
    return {"key": env["WB2A_API_KEY"], "base": "http://127.0.0.1:8788/v1",
            "model": "deepseek-v4.1-flash"}


def load_deepseek() -> dict:
    """官方渠道；若 402 自动降级 wb2a-hub。"""
    d = json.loads(_LAB_CFG.read_text(encoding="utf-8-sig"))
    src = next(s for s in d["provider_sources"] if s["id"] == "deepseek")
    prov = next(p for p in d["provider"] if p["provider_source_id"] == "deepseek")
    return {"key": src["key"][0], "base": src["api_base"],
            "model": prov["model"]}


class RealProvider:
    """DeepSeek OpenAI 兼容 provider，接口对齐 astrbot Provider.text_chat。"""

    def __init__(self, key: str, base: str, model: str):
        self.key, self.base, self.model = key, base, model
        self.calls = 0

    async def text_chat(self, prompt=None, session_id=None, system_prompt=None,
                        contexts=None, image_urls=None, **kw):
        self.calls += 1
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt or ""})
        body = {"model": self.model, "messages": messages, "max_tokens": 2000}
        last_err: Exception | None = None
        for attempt in range(3):                     # 上游不稳：5xx/超时退避重试
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(
                            f"{self.base}/chat/completions",
                            headers={"Authorization": f"Bearer {self.key}"},
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=180)) as r:
                        data = await r.json()
                        if r.status == 200:
                            return type("R", (), {"completion_text":
                                data["choices"][0]["message"]["content"]})()
                        last_err = RuntimeError(
                            f"API {r.status}: {json.dumps(data, ensure_ascii=False)[:200]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_err = e
            if attempt < 2:
                await asyncio.sleep(8 * (attempt + 1))   # 上游 retry_after 建议 10s 量级
        raise last_err


class _DictConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


async def main() -> None:
    from astrbot_plugin_promptlab.core import curator
    from astrbot_plugin_promptlab.core.engine import AdversarialEngine
    from astrbot_plugin_promptlab.core.state import SessionStore, TrainingSession

    try:
        ds = load_deepseek()
    except Exception:
        ds = _load_wb2a()
    # 预检官方渠道：402 余额不足直接切 wb2a-hub（免费 cn 通道）
    try:
        provider = RealProvider(ds["key"], ds["base"], ds["model"])
        await provider.text_chat(prompt="回复：ok", system_prompt=None)
    except RuntimeError as e:
        if "402" in str(e):
            print("官方 DeepSeek 402 没余额，切 wb2a-hub 免费通道")
            ds = _load_wb2a()
            provider = RealProvider(ds["key"], ds["base"], ds["model"])
        else:
            raise
    print(f"模型：{provider.model} @ {provider.base}")

    # 真人设（模拟被训 bot：一个书面腔明显的助手）
    persona = ("你叫小蓝，是一个QQ群机器人助手。你的职责是解答群友的问题，"
               "提供准确、规范、有礼貌的帮助。回复应当条理清晰、语言规范。")

    requirement = "回复要更像真人，短一点，不要书面腔，句尾不要句号，可以用梗"

    # 真语料挑选
    entries = curator.load_corpus(r"D:\qqmind-plugin\corpus\human_chat_corpus_1031.json")
    samples, stats = curator.curate(entries, requirement, max_lines=12)
    print(f"语料：读 {len(entries)} 挑 {stats['picked']} 条（{stats['chars']} 字）")

    tmp = tempfile.mkdtemp(prefix="promptlab_real_")
    store = SessionStore(tmp)

    class Plugin:
        def cfg(self, k, d):
            return {"max_rounds": 3, "pass_score": 70}.get(k, d)

    plugin = Plugin()
    plugin.store = store
    plugin.context = type("Ctx", (), {
        "get_using_provider": staticmethod(lambda: provider),
        "get_provider_by_id": staticmethod(lambda pid: provider),
    })()
    engine = AdversarialEngine(plugin)

    notified: list[str] = []

    async def notify(s, text):
        notified.append(text)
        print(f"\n>>> 汇报：{text[:300]}")

    s = TrainingSession(tid="real-e2e", owner=0, origin="", requirement=requirement,
                        max_rounds=3, pass_score=70, samples=samples,
                        base_prompt=persona)
    s.cur_prompt = persona
    store.save(s)

    # 单进程顺序跑（不走 create_task，报错直接抛出来看）
    engine.plugin = plugin
    await engine._run(s, notify)

    print("\n========== 实测结果 ==========")
    print("状态:", s.status, "| 轮数:", len(s.rounds), "/", s.max_rounds)
    print("清单:")
    for c in s.checklist:
        print(f"  [{c.get('weight', '中')}] {c.get('item', '')}")
    print("\n基线回复（书面腔人设）:", s.baseline_reply[:200])
    for rnd in s.rounds:
        j = rnd["judge"]
        print(f"\n-- 第{rnd['n']}轮 评分 {j['score']} verdict={j['verdict']} "
              f"diverged={j['diverged']} faithful={j['faithful']}")
        print("   候选回复:", rnd["reply"][:200])
        print("   总评:", j.get("summary", "")[:150])
        for fb in j.get("feedback", []):
            print("   反馈:", fb[:150])
    print("\n最终提示词 vs 基线：", "有变化" if s.cur_prompt != s.base_prompt
          else "无变化（异常）")
    print("优化师调用次数:", provider.calls)
    print("\n最终提示词：\n" + s.cur_prompt)


if __name__ == "__main__":
    asyncio.run(main())

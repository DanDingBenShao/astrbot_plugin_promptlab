"""端到端测试基础设施（沿用 qqmind 套路）。

不启动 AstrBot 主程序：直接构造插件实例（PromptLabPlugin），喂假事件，
用脚本 Provider（ScriptProvider）固定优化师/被训者/验收官输出。
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 让实验室的 AstrBot 可 import（只 import 类型，不启动主程序）
_ASTRBOT_ROOT = Path(r"C:\Users\16535\Documents\deepseek-harness"
                     r"\default-workspace\qqbot-lab\astrbot\AstrBot-master")
if str(_ASTRBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ASTRBOT_ROOT))
os.environ.setdefault("ASTRBOT_ROOT",
                      str(Path(tempfile.gettempdir()) / "promptlab_e2e_root"))
os.makedirs(os.environ["ASTRBOT_ROOT"], exist_ok=True)

_PLUGIN_DIR = Path(r"D:\promptlab-plugin")
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

from astrbot.api.star import Context, Star  # noqa: E402
from astrbot.core.provider.entities import LLMResponse  # noqa: E402


class ScriptProvider:
    """按 (system, user) 特征返回固定输出的假 provider。

    routes: list of (sys_contains, user_contains, output)
    依次匹配首个命中的；全不中返回 fallback。
    """

    def __init__(self, routes: list[tuple[str, str, str]],
                 fallback: str = "默认回复"):
        self.routes = routes
        self.fallback = fallback
        self.calls: list[tuple[str, str]] = []

    async def text_chat(self, prompt=None, session_id=None, system_prompt=None,
                        contexts=None, image_urls=None, **kw):
        self.calls.append((system_prompt or "", prompt or ""))
        for sys_kw, user_kw, out in self.routes:
            if sys_kw in (system_prompt or "") and user_kw in (prompt or ""):
                return LLMResponse(role="assistant", completion_text=out)
        return LLMResponse(role="assistant", completion_text=self.fallback)

    @property
    def call_count(self) -> int:
        return len(self.calls)


class DictPersonaManager:
    """get_default_persona_v3 返回固定人设。"""

    def __init__(self, prompt: str):
        self._prompt = prompt
        self.default_persona = "默认"

    async def get_default_persona_v3(self):
        return {"prompt": self._prompt, "name": "默认",
                "begin_dialogs": [], "mood_imitation_dialogs": [],
                "tools": None, "skills": None,
                "custom_error_message": None,
                "_begin_dialogs_processed": [],
                "_mood_imitation_dialogs_processed": ""}

    async def get_all_personas(self):
        return []


class FakeDB:
    def __init__(self, convs=None):
        self._convs = convs or []

    async def get_conversations(self, user_id=None, platform_id=None):
        return self._convs


class FakeContext(Context):
    """最小 Context 桩：只覆盖插件用到的路径。"""

    def __init__(self, provider: ScriptProvider, persona_prompt: str = "",
                 convs=None):
        self._provider = provider
        self.persona_manager = DictPersonaManager(persona_prompt)
        self._db = FakeDB(convs)
        self.sent: list[tuple[str, str]] = []   # (origin, text)

    def get_using_provider(self):
        return self._provider

    def get_provider_by_id(self, pid):
        return self._provider if pid else None

    def get_db(self):
        return self._db

    async def send_message(self, origin, chain):
        # MessageChain 内部持 .chain（list[dict]）；也兼容裸 list
        segs = getattr(chain, "chain", chain)
        texts = [seg.get("text", "") for seg in segs if isinstance(seg, dict)]
        self.sent.append((origin, "".join(texts)))


class _DictConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def make_event(text: str, uid: str = "10001",
               origin: str = "aiocqhttp:GroupMessage:12345") -> object:
    """构造最小可用事件桩（与管线剥离后的 message_str 语义对齐）。"""
    from types import SimpleNamespace

    ev = SimpleNamespace()
    ev.message_str = text
    ev.unified_msg_origin = origin
    ev.get_sender_id = lambda: uid
    ev.role = "member"
    _results: list[str] = []

    def plain_result(t: str):
        _results.append(t)
        return t

    ev.plain_result = plain_result
    ev._results = _results
    return ev


class PluginHarness:
    """装配完整插件（真 SessionStore 落盘到临时目录）。"""

    def __init__(self, routes: list[tuple[str, str, str]],
                 persona_prompt: str = "你是测试小助手。", convs=None,
                 cfg: dict | None = None):
        self.tmp = tempfile.mkdtemp(prefix="promptlab_test_")
        self.provider = ScriptProvider(routes)
        self.context = FakeContext(self.provider, persona_prompt, convs)
        import importlib
        import astrbot_plugin_promptlab.main as main_mod
        importlib.reload(main_mod)
        self.plugin = main_mod.PromptLabPlugin(self.context, _DictConfig(cfg or {}))
        # 数据根指到临时目录（重载后 _plugin_data_root 会用 ASTRBOT_ROOT）
        self.plugin._data_root = os.path.join(self.tmp, "pdata")
        self.plugin.store = main_mod.SessionStore(self.plugin._data_root)
        self.plugin.engine = main_mod.AdversarialEngine(self.plugin)

    async def run_cmd(self, text: str, uid: str = "10001"):
        """跑一次 /训练… 指令（按子指令分发到对应 handler）。"""
        ev = make_event(text, uid=uid)
        stripped = text.lstrip("/")
        handler_name = "cmd_train_root"
        for sub, suffix in (("训练开始", "start"), ("训练状态", "status"),
                            ("训练停", "stop"), ("训练报告", "report"),
                            ("训练继续", "resume"), ("训练样本", "samples"),
                            ("训练语料", "corpus"), ("训练列表", "list")):
            if stripped == sub or stripped.startswith(sub + " "):
                handler_name = "cmd_train_" + suffix
                break
        agen = getattr(self.plugin, handler_name)(ev)
        outs = []
        async for r in agen:
            outs.append(r)
        return outs

    def cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

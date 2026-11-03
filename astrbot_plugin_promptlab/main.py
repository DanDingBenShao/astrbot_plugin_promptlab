"""PromptLab 双AI对抗训练室 —— AstrBot 插件。

定位：提示词层对抗迭代（第一个散件，独立于 qqmind 可用）。
  人类需求 → 验收官拆清单 → [优化师改提示词 → 被训者产回复 → 验收官三轴评分]
  → 通过收敛 / 满轮必停（默认 3 轮）。

设计约束（市场规范 + 兼容性）：
- 不注册任何 on_llm_request / on_llm_response 钩子：不碰 req.contexts /
  system_prompt，对其他插件零干扰
- LLM 调用只走 provider.text_chat（官方接口）
- 会话历史读取只走 context.get_db().get_conversations()（官方接口）
- 数据全在 data/plugin_data/astrbot_plugin_promptlab/
- 日志只 from astrbot.api import logger
"""
from __future__ import annotations

import json
import time

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, MessageChain, filter

from .core.engine import AdversarialEngine
from .core.state import SessionStore, TrainingSession, _safe_int_id

PLUGIN_NAME = "astrbot_plugin_promptlab"
PLUGIN_VERSION = "0.1.0"

_CMD = "训练"


def _plugin_data_root() -> str:
    try:
        from astrbot.core.utils.astrbot_path import get_astrbot_data_path
        root = f"{get_astrbot_data_path()}/plugin_data/{PLUGIN_NAME}"
    except Exception:
        root = f"data/plugin_data/{PLUGIN_NAME}"
    import os
    os.makedirs(root, exist_ok=True)
    return root


def _arg_after(event: AstrMessageEvent, cmd: str) -> str:
    """取指令后的参数：兼容管线已剥指令头 / 原始未剥两种形态。"""
    t = (getattr(event, "message_str", "") or "").strip()
    if t.startswith("/"):
        t = t[1:].strip()
    if t == cmd:
        return ""
    if t.startswith(cmd + " "):
        return t[len(cmd) + 1:].strip()
    return t


class PromptLabPlugin(star.Star):
    """双AI对抗训练室：优化师改提示词，验收官验回复，基线对照防敷衍。"""

    def __init__(self, context: star.Context, config=None):
        super().__init__(context)
        self.plugin_config = config
        self._data_root = _plugin_data_root()
        self.store = SessionStore(self._data_root)
        self.engine = AdversarialEngine(self)
        self._manual_samples: dict[int, str] = {}   # owner -> 下次训练用的风格样本

    def cfg(self, key: str, default):
        try:
            return self.plugin_config.get(key, default)
        except Exception:
            return default

    async def terminate(self):
        """插件卸载/重载：协作停掉在跑的训练（会话已落盘，可 /训练 继续）。"""
        for owner, task in list(self.engine._tasks.items()):
            if not task.done():
                task.cancel()
        tasks = [t for t in self.engine._tasks.values()
                 if not t.done() and t.cancelled() is False]
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self.engine._tasks.clear()

    # ---------- 指令 ----------
    # 每个子指令独立注册（AstrBot 指令按「完整指令名+空格或结尾」匹配，
    # /训练开始 匹配不到 /训练 这根，所以不能靠根指令解析子命令）。
    @filter.command("训练")
    async def cmd_train_root(self, event: AstrMessageEvent):
        """训练室：无参数=帮助。"""
        arg = _arg_after(event, "训练")
        if not arg or arg in ("帮助", "help", "?", "？"):
            yield event.plain_result(self._help())
            return
        yield event.plain_result(
            "未知子命令：" + arg.split()[0] + "。" + self._help())

    @filter.command("训练开始")
    async def cmd_train_start(self, event: AstrMessageEvent):
        arg = _arg_after(event, "训练开始")
        async for r in self._start(event, _safe_int_id(event.get_sender_id()),
                                   arg):
            yield r

    @filter.command("训练状态")
    async def cmd_train_status(self, event: AstrMessageEvent):
        yield event.plain_result(self._status(_safe_int_id(event.get_sender_id())))

    @filter.command("训练停")
    async def cmd_train_stop(self, event: AstrMessageEvent):
        yield event.plain_result(self._stop(_safe_int_id(event.get_sender_id())))

    @filter.command("训练报告")
    async def cmd_train_report(self, event: AstrMessageEvent):
        yield event.plain_result(self._report(_safe_int_id(event.get_sender_id())))

    @filter.command("训练继续")
    async def cmd_train_resume(self, event: AstrMessageEvent):
        arg = _arg_after(event, "训练继续")
        async for r in self._resume(event, _safe_int_id(event.get_sender_id()),
                                    arg):
            yield r

    @filter.command("训练样本")
    async def cmd_train_samples(self, event: AstrMessageEvent):
        arg = _arg_after(event, "训练样本")
        yield event.plain_result(
            self._set_samples(_safe_int_id(event.get_sender_id()), arg))

    @filter.command("训练列表")
    async def cmd_train_list(self, event: AstrMessageEvent):
        yield event.plain_result(self._list(_safe_int_id(event.get_sender_id())))

    # ---------- 子命令实现 ----------
    async def _start(self, event: AstrMessageEvent, uid: int, req: str):
        if self.engine.busy(uid):
            yield event.plain_result("已有训练在跑，先 /训练停 或等它结束。")
            return
        if not req:
            yield event.plain_result(
                "用法：/训练开始 <你的需求>\n例：/训练开始 回复要更像真人，"
                "别书面腔，句尾别带句号，可以用梗")
            return
        # 基线提示词：当前默认人设（只读，不修改）
        base_prompt = await self._base_prompt()
        if not base_prompt:
            yield event.plain_result(
                "拿不到当前人设提示词（未配置默认人设？）。"
                "可先用 /训练样本 <风格参考文本> 提供素材后再试。")
            return
        samples = self._manual_samples.pop(uid, "") or await self._collect_samples(uid)
        max_rounds = max(1, min(10, int(self.cfg("max_rounds", 3) or 3)))
        pass_score = max(1, min(100, int(self.cfg("pass_score", 70) or 70)))
        tid = time.strftime("%m%d-%H%M%S")
        s = TrainingSession(tid=tid, owner=uid,
                            origin=event.unified_msg_origin,
                            requirement=req, max_rounds=max_rounds,
                            pass_score=pass_score, samples=samples,
                            base_prompt=base_prompt)
        s.cur_prompt = base_prompt
        self.store.save(s)
        ok = self.engine.launch(s, self._notify)
        if not ok:
            yield event.plain_result("启动失败：已有同名任务在跑。")
            return
        yield event.plain_result(
            f"训练 #{tid} 已开跑（最多 {max_rounds} 轮，及格线 {pass_score} 分）\n"
            f"流程：拆清单 → 基线 → 对抗轮次 → 必停交付\n"
            f"每轮结束自动汇报进度；跑完用 /训练报告 看完整报告。"
            + ("（风格样本：已从会话历史取到）" if samples
               else "（无风格样本，纯按需求训练）"))

    def _status(self, uid: int) -> str:
        s = self.store.latest_for_owner(uid)
        if not s:
            return "还没有训练记录。/训练开始 <需求> 开一个。"
        if self.engine.busy(uid):
            head = f"训练 #{s.tid} 正在跑（{len(s.rounds)}/{s.max_rounds} 轮）"
        else:
            head = f"训练 #{s.tid} 状态：{s.status}"
        lines = [head, f"需求：{s.requirement[:80]}"]
        if s.checklist:
            lines.append("清单：" + "；".join(
                f"[{c.get('weight', '中')}] {c.get('item', '')}"
                for c in s.checklist[:4]))
        last = s.last_round()
        if last:
            lines.append(f"末轮评分：{last['judge']['score']} 分"
                         f"（{last['judge']['verdict']}）")
            lines.append("  " + _scores_brief(last["judge"]))
        elif s.baseline_reply:
            lines.append("基线已就绪，对抗轮次进行中。")
        return "\n".join(lines)

    def _stop(self, uid: int) -> str:
        if self.engine.request_stop(uid):
            return "已请求停止：本轮跑完后停（不硬杀，免得白花已用的调用）。"
        return "当前没有在跑的训练。"

    def _report(self, uid: int) -> str:
        s = self.store.latest_for_owner(uid)
        if not s:
            return "还没有训练记录。"
        if s.status in (TrainingSession.STATUS_RUNNING,) or self.engine.busy(uid):
            return f"训练 #{s.tid} 还在跑（{len(s.rounds)}/{s.max_rounds}），跑完再来看报告。"
        if not s.rounds:
            return f"训练 #{s.tid} 没有完成的轮次（状态 {s.status}）。"
        lines = [f"—— 训练 #{s.tid} 报告 ——"]
        passed = s.status == TrainingSession.STATUS_PASSED
        lines.append(f"结果：{'✅ 通过' if passed else '⛔ 未通过（满轮停止）'}，"
                     f"用 {len(s.rounds)}/{s.max_rounds} 轮")
        last = s.last_round()
        if last:
            j = last["judge"]
            lines.append(f"末轮评分：{j['score']} 分 | 变化度"
                         f"{'足够' if j['diverged'] else '不足'} | 保真"
                         f"{'完好' if j['faithful'] else '崩坏'}")
            lines.append("逐项：")
            lines.append(_scores_detail(j))
            if j["summary"]:
                lines.append(f"总评：{j['summary']}")
        lines.append("")
        lines.append("—— 收敛后的提示词（自行粘贴回 bot 人设）——")
        lines.append(s.cur_prompt)
        return "\n".join(lines)

    async def _resume(self, event: AstrMessageEvent, uid: int, rest: str):
        if self.engine.busy(uid):
            yield event.plain_result("已有训练在跑。")
            return
        s = self.store.latest_for_owner(uid)
        if not s:
            yield event.plain_result("没有可继续的训练。")
            return
        if s.status == TrainingSession.STATUS_PASSED:
            yield event.plain_result(
                f"训练 #{s.tid} 已通过收敛，无需继续。要再训就 /训练开始 <新需求>。")
            return
        # 继续 N：加轮数（满轮停止的续命入口）
        if rest.isdigit():
            extra = max(1, min(10, int(rest)))
            s.max_rounds = min(10, s.max_rounds + extra)
        if s.rounds_left() <= 0:
            yield event.plain_result(
                f"训练 #{s.tid} 轮次已用完（{s.max_rounds} 轮）。"
                f"用「/训练继续 2」加 2 轮再跑。")
            return
        s.status = TrainingSession.STATUS_PENDING
        self.store.save(s)
        ok = self.engine.launch(s, self._notify)
        yield event.plain_result(
            f"训练 #{s.tid} 继续，剩余 {s.rounds_left()} 轮。" if ok
            else "启动失败。")

    def _list(self, uid: int) -> str:
        rows = []
        for tid in reversed(self.store.list_tids()[-8:]):
            s = self.store.load(tid)
            if s and s.owner == uid:
                rows.append(f"#{tid}  {s.status}  {len(s.rounds)}轮  "
                            f"{s.requirement[:24]}")
        if not rows:
            return "该用户没有训练记录。"
        return "最近的训练：\n" + "\n".join(rows)

    def _set_samples(self, uid: int, text: str) -> str:
        if not text:
            self._manual_samples.pop(uid, None)
            return "已清除你手动准备的样本（下次训练自动从会话历史取）。"
        self._manual_samples[uid] = text[:2000]
        return f"样本已就绪（{len(text)} 字），下次 /训练开始 时生效。"

    # ---------- 支撑 ----------
    async def _base_prompt(self) -> str:
        """当前默认人设提示词（只读）。多种 persona API 形态逐个尝试。"""
        pm = getattr(self.context, "persona_manager", None)
        if pm is None:
            return ""
        try:
            p = await pm.get_default_persona_v3()
            if isinstance(p, dict) and p.get("prompt"):
                return str(p["prompt"])
        except Exception as e:
            logger.warning(f"[promptlab] get_default_persona_v3 failed: {e!r}")
        try:
            all_p = await pm.get_all_personas()
            default_name = getattr(pm, "default_persona", "")
            for it in (all_p or []):
                d = it if isinstance(it, dict) else getattr(it, "__dict__", {})
                if d.get("name") == default_name and d.get("prompt"):
                    return str(d["prompt"])
            for it in (all_p or []):
                d = it if isinstance(it, dict) else getattr(it, "__dict__", {})
                if d.get("prompt"):
                    return str(d["prompt"])
        except Exception as e:
            logger.warning(f"[promptlab] get_all_personas failed: {e!r}")
        return ""

    async def _collect_samples(self, uid: int) -> str:
        """从官方会话历史取该用户最近的对话做风格样本（只读）。"""
        limit = int(self.cfg("samples_limit", 20) or 0)
        chars = int(self.cfg("samples_chars", 1200) or 1200)
        if limit <= 0:
            return ""
        try:
            db = self.context.get_db()
            convs = await db.get_conversations(user_id=str(uid))
        except Exception as e:
            logger.warning(f"[promptlab] collect samples failed: {e!r}")
            return ""
        msgs: list[tuple[str, str]] = []
        for c in convs or []:
            content = getattr(c, "content", None)
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except ValueError:
                    continue
            if not isinstance(content, list):
                continue
            for m in content:
                if not isinstance(m, dict):
                    continue
                role, text = m.get("role"), m.get("content")
                if isinstance(text, list):
                    text = "".join(
                        p.get("text", "") for p in text
                        if isinstance(p, dict) and p.get("type") == "text")
                if role in ("user", "assistant") and isinstance(text, str) \
                        and text.strip():
                    msgs.append(("用户" if role == "user" else "Bot", text.strip()))
            if len(msgs) >= limit * 2:
                break
        if not msgs:
            return ""
        lines = [f"{r}：{t}" for r, t in msgs[-limit:]]
        return ("\n".join(lines))[:chars]

    async def _notify(self, s: TrainingSession, text: str) -> None:
        """训练进行中/结束的主动汇报（走官方 send_message）。"""
        origin = s.origin
        if not origin:
            return
        try:
            await self.context.send_message(origin, MessageChain(
                [{"type": "Plain", "text": text}]))
        except Exception as e:
            logger.warning(f"[promptlab] notify to {origin} failed: {e!r}")

    def _help(self) -> str:
        return (
            f"PromptLab 双AI对抗训练室 v{PLUGIN_VERSION}\n"
            "你提需求 → 优化师AI改提示词 → 被训AI产回复 → 验收官AI验收，"
            "不通过就再来一轮，默认 3 轮必停。\n"
            f"/{_CMD}开始 <需求>   开一次训练\n"
            f"/{_CMD}状态          看进度\n"
            f"/{_CMD}停            本轮结束后停下\n"
            f"/{_CMD}报告          完整报告+收敛提示词\n"
            f"/{_CMD}继续 [N]      继续/加 N 轮\n"
            f"/{_CMD}样本 <文本>   手动给风格样本（可选）\n"
            f"/{_CMD}列表          历史训练")


def _scores_brief(j: dict) -> str:
    parts = []
    for sc in j.get("scores", [])[:4]:
        parts.append(("✅" if sc.get("pass") else "❌") + sc.get("item", "")[:14])
    return " ".join(parts)


def _scores_detail(j: dict) -> str:
    lines = []
    for sc in j.get("scores", []):
        mark = "✅" if sc.get("pass") else "❌"
        ev = sc.get("evidence", "")
        lines.append(f"  {mark}[{sc.get('weight', '中')}] {sc.get('item', '')}"
                     + (f"——{ev}" if ev else ""))
    return "\n".join(lines)

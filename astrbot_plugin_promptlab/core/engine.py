"""PromptLab 对抗训练引擎：状态机 + LLM 调用编排。

流程：
  立项 → 清单拆解 → 基线（当前提示词产一份回复）
       → 循环[优化师改提示词 → 被训者产回复 → 验收官评分]
       → 通过即收敛 / 满轮必停

所有 LLM 调用走 provider.text_chat（官方 Provider 接口），
不碰任何会话上下文（req.contexts）——纯离线候选迭代，市场审查零风险。
每轮结束即落盘（原子写），崩溃最多丢「正在跑的那一轮」。
"""
from __future__ import annotations

import asyncio
import time
import uuid

from astrbot.api import logger

from .judge import (DEFAULT_PROBE, build_act_user, build_checklist_prompt,
                    build_judge_prompt, build_optimizer_prompt,
                    parse_checklist, parse_judge)
from .state import TrainingSession


def _pick_provider(context, model_id: str):
    """按配置选 provider：指定 id → get_provider_by_id；空 → 主 provider。"""
    if model_id:
        p = context.get_provider_by_id(model_id)
        if p is None:
            raise RuntimeError(f"配置的模型 id 不存在：{model_id}")
        return p
    return context.get_using_provider()


async def _chat(provider, system: str, user: str,
                max_tokens: int = 1500, temperature: float = 0.5) -> str:
    """单次 LLM 调用，返回纯文本。失败抛异常由调用方兜。"""
    out = await provider.text_chat(prompt=user, session_id=None,
                                   system_prompt=system, contexts=[],
                                   image_urls=[])
    return (out.completion_text or "").strip()


def _fmt_scores(scores: list[dict]) -> str:
    lines = []
    for sc in scores:
        mark = "✅" if sc.get("pass") else "❌"
        w = sc.get("weight", "中")
        ev = sc.get("evidence", "")
        lines.append(f"  {mark}[{w}] {sc.get('item', '')}"
                     + (f"——{ev}" if ev else ""))
    return "\n".join(lines) or "  （验收官未给出逐项明细）"


class AdversarialEngine:
    """对抗训练状态机。一个插件实例持一个；同一时刻同 owner 只跑一个任务。"""

    def __init__(self, plugin):
        self.plugin = plugin
        self._tasks: dict[int, asyncio.Task] = {}   # owner -> task（强引用防 GC）
        self._cancel: set[int] = set()

    # ---------- 对外入口 ----------
    def launch(self, s: TrainingSession, notify) -> bool:
        """后台跑一轮训练。notify(session, text) 用于主动汇报。同 owner 已有任务在跑则拒绝。"""
        owner = s.owner
        if owner in self._tasks and not self._tasks[owner].done():
            return False
        self._cancel.discard(owner)
        self._tasks[owner] = asyncio.create_task(self._run(s, notify))
        return True

    def request_stop(self, owner: int) -> bool:
        """请求在当前轮结束后停下（协作式，不硬杀：硬杀会丢本轮已花的调用）。"""
        if owner in self._tasks and not self._tasks[owner].done():
            self._cancel.add(owner)
            return True
        return False

    def busy(self, owner: int) -> bool:
        t = self._tasks.get(owner)
        return bool(t and not t.done())

    # ---------- 内部：完整训练流程 ----------
    async def _run(self, s: TrainingSession, notify) -> None:
        cfg = self.plugin.cfg
        store = self.plugin.store
        ctx = self.plugin.context
        try:
            # -- 0. 清单拆解（验收官）
            s.status = TrainingSession.STATUS_RUNNING
            store.save(s)
            if not s.checklist:
                judge_p = _pick_provider(ctx, cfg("judge_model", ""))
                try:
                    text = await _chat(
                        judge_p, "你是验收官。",
                        build_checklist_prompt(s.requirement),
                        max_tokens=600, temperature=0.2)
                    s.checklist = parse_checklist(text)
                except Exception as e:
                    logger.warning(f"[promptlab] checklist failed, fallback: {e!r}")
                    s.checklist = [{"item": "回复整体更符合人类需求", "weight": "高"}]
                store.save(s)

            # -- 1. 基线回复（只跑一次）
            if not s.baseline_reply:
                s.baseline_reply = await self._act(s, s.base_prompt)
                store.save(s)

            # -- 2. 轮次循环
            while s.rounds_left() > 0:
                if s.owner in self._cancel:
                    self._cancel.discard(s.owner)
                    s.status = TrainingSession.STATUS_STOPPED
                    store.save(s)
                    await self._notify_done(s, notify, "已按你的要求停止。")
                    return
                rnd = await self._one_round(s)
                store.save(s)
                if rnd["judge"]["verdict"] == "pass":
                    s.status = TrainingSession.STATUS_PASSED
                    break
                if s.rounds_left() <= 0:
                    s.status = TrainingSession.STATUS_MAXED
                    break
            else:
                s.status = TrainingSession.STATUS_MAXED
            store.save(s)
            await self._finish(s, notify)
        except asyncio.CancelledError:
            s.status = TrainingSession.STATUS_STOPPED
            store.save(s)
            raise
        except Exception as e:
            logger.error(f"[promptlab] training task failed: {e!r}")
            s.status = TrainingSession.STATUS_FAILED
            s.report_msg = f"训练失败：{e!r}"
            store.save(s)
            try:
                await self._notify_done(s, notify, s.report_msg)
            except Exception:
                pass

    async def _one_round(self, s: TrainingSession) -> dict:
        """一轮 = 优化师改提示词 → 被训者产回复 → 验收官评分。"""
        n = len(s.rounds) + 1
        cfg = self.plugin.cfg
        ctx = self.plugin.context
        # 2a. 优化师出候选提示词
        last = s.last_round()
        feedback = last["judge"]["feedback"] if last and last.get("judge") \
            else ["首轮：按人类需求与清单，直接给出修改后的提示词"]
        opt_p = _pick_provider(ctx, cfg("prompt_model", ""))
        cur = s.cur_prompt if s.rounds else s.base_prompt
        opt_user = build_optimizer_prompt(s.requirement, s.checklist,
                                          cur, feedback, s.base_prompt)
        try:
            new_prompt = (await _chat(opt_p, "你是提示词优化师。", opt_user)).strip()
        except Exception as e:
            raise RuntimeError(f"优化师调用失败（第{n}轮）：{e!r}") from e
        if not new_prompt:
            raise RuntimeError(f"优化师返回空提示词（第{n}轮）")
        # 2b. 被训者用候选提示词产回复
        reply = await self._act(s, new_prompt)
        # 2c. 验收官评分
        judge_p = _pick_provider(ctx, cfg("judge_model", ""))
        j_user = build_judge_prompt(s.requirement, s.checklist,
                                    s.baseline_reply, reply)
        judge = None
        for _ in range(2):   # JSON 解析失败重试一次
            try:
                jtext = await _chat(judge_p, "你是严格但公正的验收官。", j_user,
                                    temperature=0.2)
            except Exception as e:
                raise RuntimeError(f"验收官调用失败（第{n}轮）：{e!r}") from e
            judge = parse_judge(jtext)
            if judge:
                break
        if judge is None:
            raise RuntimeError(f"验收官输出无法解析（第{n}轮）：{jtext[:120]!r}")
        # 基线一致死锁：验收官漏判时本地兜底——与基线几乎相同直接判 fail
        if judge["verdict"] == "pass" and s.baseline_reply and \
                _too_similar(reply, s.baseline_reply):
            judge["verdict"] = "fail"
            judge["score"] = min(judge["score"], 45)
            judge["feedback"] = ["与基线回复几乎相同（本地检测）：需要实质改变表达方式，不是改几个词"]
            judge["diverged"] = False
        rnd = {"n": n, "prompt": new_prompt, "reply": reply,
               "judge": judge, "ts": time.time()}
        s.append_round(rnd)
        s.cur_prompt = new_prompt
        return rnd

    async def _act(self, s: TrainingSession, prompt: str) -> str:
        """被训者：system=候选提示词，产一段回复。"""
        cfg = self.plugin.cfg
        act_p = _pick_provider(self.plugin.context, cfg("act_model", ""))
        scenario = f"【风格参照（真实对话节选，供感受语气）】\n{s.samples}" \
            if s.samples else ""
        user = build_act_user(scenario, _default_probe(s))
        try:
            text = await _chat(act_p, prompt, user)
        except Exception as e:
            raise RuntimeError(f"被训者调用失败：{e!r}") from e
        if not text:
            raise RuntimeError("被训者返回空回复")
        return text[:2000]

    async def _finish(self, s: TrainingSession, notify) -> None:
        """收敛：写最终报告。"""
        last = s.last_round()
        passed = s.status == TrainingSession.STATUS_PASSED
        s.final_report = {
            "status": s.status,
            "rounds_used": len(s.rounds),
            "max_rounds": s.max_rounds,
            "score": last["judge"]["score"] if last else 0,
            "summary": last["judge"]["summary"] if last else "",
        }
        lines = [f"训练结束（{'✅ 通过' if passed else '⛔ 满轮停止'}，"
                 f"用时 {len(s.rounds)}/{s.max_rounds} 轮，"
                 f"末轮评分 {s.final_report['score']}）"]
        if s.final_report["summary"]:
            lines.append(f"验收官总评：{s.final_report['summary']}")
        lines.append("")
        lines.append("—— 最终提示词（请自行粘贴回 bot 人设）——")
        lines.append(s.cur_prompt)
        s.report_msg = "\n".join(lines)
        store = self.plugin.store
        store.save(s)
        await self._notify_done(s, notify, s.report_msg)

    async def _notify_done(self, s: TrainingSession, notify, text: str) -> None:
        try:
            await notify(s, text)
        except Exception as e:
            logger.warning(f"[promptlab] notify failed: {e!r}")


def _default_probe(s: TrainingSession) -> str:
    """从需求里抽一句当探针；抽不出就用默认。"""
    req = (s.requirement or "").strip()
    if req:
        line = req.splitlines()[0].strip()
        return line[:120] if line else DEFAULT_PROBE
    return DEFAULT_PROBE


def _too_similar(a: str, b: str) -> bool:
    """本地兜底：与基线的重合度过高判「变化不足」。
    完全相同永远算；包含关系（≥6字整段复现）算；逐字重合率>0.9（≥12字）算。"""
    na = "".join(a.split())
    nb = "".join(b.split())
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long_ = (na, nb) if len(na) <= len(nb) else (nb, na)
    if 6 <= len(short) and short in long_:
        return True
    if len(na) < 12 or len(nb) < 12:
        return False
    same = sum(1 for x, y in zip(na, nb) if x == y)
    return same / max(len(na), len(nb)) > 0.9

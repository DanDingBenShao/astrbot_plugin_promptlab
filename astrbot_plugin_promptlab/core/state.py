"""PromptLab 数据层：会话状态 + 原子落盘。

每轮结束即写盘（原子替换：tmp + os.replace），崩溃最多丢「正在跑的那一轮」，
已完成轮次与上一份提示词永远安全。数据全在 data/plugin_data/astrbot_plugin_promptlab/。
"""
from __future__ import annotations

import json
import os
import time


def _safe_int_id(v) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return 0


def _json_write_atomic(path: str, obj) -> None:
    """原子写 JSON：先写 tmp 再 os.replace，崩溃不会留半截文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _json_read(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class TrainingSession:
    """一次对抗训练的完整状态。字段即落盘格式。"""

    STATUS_PENDING = "pending"      # 已立项，未跑基线
    STATUS_RUNNING = "running"      # 正在跑轮次
    STATUS_PASSED = "passed"        # 验收通过，收敛
    STATUS_MAXED = "maxed"          # 满轮必停
    STATUS_FAILED = "failed"        # 系统性失败（模型报错等）
    STATUS_STOPPED = "stopped"      # 用户手动停止

    def __init__(self, tid: str, owner: int, origin: str, requirement: str,
                 max_rounds: int = 3, pass_score: int = 70,
                 samples: str = "", base_prompt: str = ""):
        self.tid = tid
        self.created = time.time()
        self.status = self.STATUS_PENDING
        self.owner = owner          # 训练发起人 QQ 号
        self.origin = origin        # 汇报目的地 unified_msg_origin
        self.requirement = requirement   # 人类需求原文
        self.checklist: list[str] = []   # 验收官拆的可检查清单
        self.samples = samples           # 风格样本（真实对话节选）
        self.base_prompt = base_prompt   # 被训 bot 当前提示词（基线）
        self.cur_prompt = base_prompt    # 当前候选提示词
        self.baseline_reply = ""         # 基线回复（用 base_prompt 产的）
        self.rounds: list[dict] = []     # [{"n","reply","scores","verdict","reasons","new_prompt"}]
        self.max_rounds = max_rounds
        self.pass_score = pass_score
        self.final_report: dict = {}     # 最终验收报告
        self.report_msg = ""             # 给人看的总结

    # ---------- 序列化 ----------
    def to_dict(self) -> dict:
        return {
            "tid": self.tid,
            "created": self.created,
            "status": self.status,
            "owner": self.owner,
            "origin": self.origin,
            "requirement": self.requirement,
            "checklist": self.checklist,
            "samples": self.samples,
            "base_prompt": self.base_prompt,
            "cur_prompt": self.cur_prompt,
            "baseline_reply": self.baseline_reply,
            "rounds": self.rounds,
            "max_rounds": self.max_rounds,
            "pass_score": self.pass_score,
            "final_report": self.final_report,
            "report_msg": self.report_msg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingSession":
        s = cls(tid=str(d.get("tid", "")),
                owner=int(d.get("owner", 0)),
                origin=str(d.get("origin", "")),
                requirement=str(d.get("requirement", "")),
                max_rounds=int(d.get("max_rounds", 3)),
                pass_score=int(d.get("pass_score", 70)))
        s.created = float(d.get("created", s.created))
        s.status = str(d.get("status", cls.STATUS_PENDING))
        s.checklist = list(d.get("checklist") or [])
        s.samples = str(d.get("samples", ""))
        s.base_prompt = str(d.get("base_prompt", ""))
        s.cur_prompt = str(d.get("cur_prompt", ""))
        s.baseline_reply = str(d.get("baseline_reply", ""))
        s.rounds = list(d.get("rounds") or [])
        s.final_report = dict(d.get("final_report") or {})
        s.report_msg = str(d.get("report_msg", ""))
        return s

    # ---------- 轮次推进 ----------
    def last_round(self) -> dict | None:
        return self.rounds[-1] if self.rounds else None

    def append_round(self, rnd: dict) -> None:
        self.rounds.append(rnd)

    def rounds_left(self) -> int:
        return max(0, self.max_rounds - len(self.rounds))


class SessionStore:
    """会话落盘存储：data/plugin_data/astrbot_plugin_promptlab/sessions/<tid>.json"""

    def __init__(self, data_root: str):
        self.sessions_dir = os.path.join(data_root, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)

    def _path(self, tid: str) -> str:
        return os.path.join(self.sessions_dir, f"{tid}.json")

    def save(self, s: TrainingSession) -> None:
        _json_write_atomic(self._path(s.tid), s.to_dict())

    def load(self, tid: str) -> TrainingSession | None:
        d = _json_read(self._path(tid))
        return TrainingSession.from_dict(d) if d else None

    def list_tids(self) -> list[str]:
        try:
            names = os.listdir(self.sessions_dir)
        except OSError:
            return []
        return sorted(n[:-5] for n in names if n.endswith(".json"))

    def latest_for_owner(self, owner: int) -> TrainingSession | None:
        """该用户最近一次训练（按创建时间，不分状态——续跑/查看都用它）。"""
        best: TrainingSession | None = None
        for tid in self.list_tids():
            s = self.load(tid)
            if s and s.owner == owner and (best is None or s.created > best.created):
                best = s
        return best

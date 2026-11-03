"""PromptLab 语料管理员：语料包 → 风格参考样本的自动挑选。

支持格式（自动识别）：
- JSON list[dict]：qqmind 画像语料（gid/time/nick/uid/text）或 {role,content}
- JSONL（每行一个 JSON）
- TXT 纯文本（一行一条）

挑选策略（全部本地规则，零 token）：
1. 清洗：丢弃命令/空消息/超长刷屏/链接/图片占位/纯标点
2. 特征匹配：需求里提到的风格词（如「短」「句尾」「梗」「口语」）命中消息文本的优先
3. 长度均衡：短/中/长三档各取一部分（真人消息本就长短不一）
4. 去重 + 保持原始顺序感
"""
from __future__ import annotations

import json
import os
import random
import re

# 清洗规则
_RE_CMD = re.compile(r"^[/!#.]|^\s*$")
_RE_LINK = re.compile(r"https?://|www\.|\.(com|cn|net|org|xyz|top)\b")
_RE_MEDIA = re.compile(
    r"^\[(图片|视频|语音|文件|动画表情|转发消息|XML|JSON|卡片|分享|"
    r"引用消息|合并转发|表情|视频号|直播)\]?|^\[CQ:")
_RE_PUNCT_ONLY = re.compile(r"^[\s\W_]+$", re.UNICODE)
# 引用消息前缀剥掉后还有正文才算有效
_RE_QUOTE = re.compile(r"^\[引用消息[^\]]*\]\s*")
# At/回复组件（qqmind 语料里组件以 [At:qq号] 形式出现，任意位置都不是正文）
_RE_AT = re.compile(r"\[At[:：][^\]]*\]")
# 内嵌媒体/表情标记（任意位置）：[表情:176] 这类，剥掉后正文还有才算
_RE_INLINE_MEDIA = re.compile(r"\[(?:表情|图片|视频|语音|文件|动画表情)[:：][^\]]*\]")

MAX_TEXT_LEN = 80        # 单条超过按刷屏丢弃
MIN_TEXT_LEN = 2

# 特征词→挑选偏好：需求里出现这些词时，命中对应特征的消息权重加倍
FEATURE_HINTS: dict[str, re.Pattern] = {
    "短": re.compile(r"^.{1,8}$"),
    "口语": re.compile(r"[哈嘿嗨哦呀嘛呗咋啥诶卧槽草乐绷6]|笑死|寄|典"),
    "梗": re.compile(r"[哈嘿草乐绷寄典6]|笑了|绷不住|乐死|无敌了"),
    "表情": re.compile(r"[😄🤣😂👍🎉🔥😭🥺✨]"),
    "句尾": re.compile(r"[^。！？．\s]$"),
    "标点": re.compile(r"[^。！？．，,\s]$"),
    "疑问": re.compile(r"[?？]\s*$"),
    "反问": re.compile(r"[?？]\s*$"),
    "语气": re.compile(r"[啊呀哦嘛呗呢吧哈嘿]"),
    "情绪": re.compile(r"！{2,}|？{2,}|哈哈|草|乐|绷|哭死|急了"),
}


def _extract_text(entry) -> tuple[str, str] | None:
    """从一条语料记录提取 (sender, text)；无法提取返回 None。"""
    if isinstance(entry, str):
        return "", entry
    if not isinstance(entry, dict):
        return None
    if "text" in entry:                       # qqmind 画像语料
        return (str(entry.get("nick") or entry.get("uid") or ""),
                str(entry.get("text") or ""))
    if "content" in entry:                    # {role, content} 格式
        c = entry.get("content")
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c
                        if isinstance(p, dict) and p.get("type") == "text")
        role = entry.get("role", "")
        if role not in ("user", "assistant"):
            return None
        return ("用户" if role == "user" else "Bot"), str(c or "")
    return None


def clean_entry(sender: str, text: str) -> str | None:
    """清洗单条，返回规整文本或 None（丢弃）。"""
    t = (text or "").strip()
    # 循环剥组件前缀：真语料里引用/At 常叠在一起（[引用…][At:…] 正文）
    while True:
        t2 = _RE_AT.sub("", t)
        t2 = _RE_QUOTE.sub("", t2)
        t2 = _RE_INLINE_MEDIA.sub("", t2)
        t2 = t2.strip()
        if t2 == t:
            break
        t = t2
    if not t or _RE_CMD.search(t) or _RE_MEDIA.match(t):
        return None
    if _RE_LINK.search(t) or _RE_PUNCT_ONLY.match(t):
        return None
    if len(t) > MAX_TEXT_LEN or len(t) < MIN_TEXT_LEN:
        return None
    return t


def load_corpus(path: str) -> list[tuple[str, str]]:
    """读语料文件 → [(sender, text)]，格式自动识别；失败返回空表。"""
    if not os.path.isfile(path):
        return []
    try:
        raw = open(path, "r", encoding="utf-8-sig").read()
    except (OSError, ValueError):
        return []
    entries: list = []
    stripped = raw.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            data = json.loads(raw)
            entries = data if isinstance(data, list) else [data]
        except ValueError:
            entries = []
    if not entries:
        # JSONL：逐行解析
        for line in raw.splitlines():
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                try:
                    entries.append(json.loads(line))
                except ValueError:
                    pass
        if not entries:
            # TXT：一行的算一条
            entries = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    out: list[tuple[str, str]] = []
    for e in entries:
        got = _extract_text(e)
        if got is None:
            continue
        sender, text = got
        t = clean_entry(sender, text)
        if t:
            out.append((sender or "?", t))
    return out


def _match_score(text: str, patterns: list[re.Pattern]) -> int:
    return sum(2 for p in patterns if p.search(text))


def curate(entries: list[tuple[str, str]], requirement: str,
           max_lines: int = 14, max_chars: int = 1200,
           seed: int | None = None) -> tuple[str, dict]:
    """从语料挑参考样本 → (样本文本, 统计)。

    策略：特征匹配加分 + 长度分档均衡 + 去重，最后保序拼接。
    """
    rng = random.Random(seed)
    pats = [p for kw, p in FEATURE_HINTS.items() if kw in (requirement or "")]
    # 打分 + 轻微随机抖动（同分不每次选同一批）
    scored = []
    seen: set[str] = set()
    for sender, text in entries:
        key = text
        if key in seen:
            continue
        seen.add(key)
        score = _match_score(text, pats) + rng.random()
        scored.append((score, len(text), sender, text))
    if not scored:
        return "", {"total": 0, "picked": 0}
    # 长度三档（短/中/长）按各自池轮转取，保证样本长短都有
    buckets: dict[str, list] = {"s": [], "m": [], "l": []}
    for score, ln, sender, text in scored:
        b = "s" if ln <= 8 else ("m" if ln <= 25 else "l")
        buckets[b].append((score, sender, text))
    for b in buckets.values():
        b.sort(key=lambda x: -x[0])
    picks: list[tuple[str, str]] = []
    order = ["s", "m", "l"] if rng.random() < 0.5 else ["m", "s", "l"]
    while len(picks) < max_lines and any(buckets.values()):
        for b in order:
            if buckets[b] and len(picks) < max_lines:
                _, sender, text = buckets[b].pop(0)
                picks.append((sender, text))
    picks.sort(key=lambda x: 0)     # 保持挑选序（打分序），不再洗牌
    lines = [f"{s}：{t}" if s not in ("?", "", "用户", "Bot")
             else t for s, t in picks]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0]
    stats = {
        "total": len(entries),
        "cleaned": len(seen),
        "picked": len(picks),
        "chars": len(text),
    }
    return text, stats


def curate_from_file(path: str, requirement: str,
                     max_lines: int = 14, max_chars: int = 1200,
                     seed: int | None = None) -> tuple[str, dict]:
    """读文件 + 挑选一步到位。文件不存在返回 ("", stats)。"""
    entries = load_corpus(path)
    if not entries:
        return "", {"total": 0, "picked": 0, "error": f"读不到有效语料：{path}"}
    return curate(entries, requirement, max_lines, max_chars, seed)

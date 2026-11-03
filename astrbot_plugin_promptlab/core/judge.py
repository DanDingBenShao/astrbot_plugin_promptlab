"""PromptLab 提示词工程层：优化师与验收官的角色提示词 + 响应解析。

对抗闭环：
  人类需求 → 清单拆解 → [优化师改提示词 → 被训者产回复 → 验收官三轴评分] ×N
  通过即收敛；满轮必停。基线对照是死锁「改完还是一样」的关键。
"""
from __future__ import annotations

import json
import re


# ============ 需求清单拆解（验收官第 0 步） ============
CHECKLIST_PROMPT = (
    "你是验收官。用户将要把一段机器人提示词改得更符合他的需求。"
    "你的第一步：把用户的需求拆成 3-6 条「可检查项」——每条必须能在一段回复文本上"
    "直接判定真假，禁止模糊表述（「更像人」不行，「句尾无句号、不用书面衔接词」可以）。\n"
    "同时给每条检查项标注权重（高/中/低）。\n"
    "只输出 JSON 数组，每项：{\"item\": \"检查项\", \"weight\": \"高|中|低\"}\n"
    "用户需求：\n{requirement}"
)

# ============ 被训者产回复 ============
# system = 候选提示词本身，user = 场景指令。场景=样本+要求。
ACT_USER_TMPL = (
    "{scenario}"
    "\n\n---\n"
    "现在请以这个身份，对下面这句话做出回复（只输出回复本身，不要任何解释、"
    "前缀、引号或括号说明）：\n"
    "{probe}"
)

# ============ 优化师：改提示词 ============
OPTIMIZER_PROMPT = (
    "你是提示词优化师。任务：根据验收反馈修改一段 bot 的系统提示词，"
    "让被训 bot 的回复更符合人类需求。\n"
    "规则：\n"
    "1. 只改与需求相关的部分，保住原有的语言、格式、人设基本盘（角色名/称呼/背景不动，"
    "除非需求明确要求改）。\n"
    "2. 验收官给的每一条不合格理由都要有对应动作；合格的方面不要动（防止改过头）。\n"
    "3. 上一轮已经加过的要求，验收官仍判不合格时，换一种写法：更具体、"
    "给正反例、或调整措辞位置（放在提示词前部），不要原样重复或堆叠同类句子。\n"
    "4. 只输出修改后的完整提示词，不要解释、不要代码块。\n"
    "\n{baseline_block}"
    "当前提示词：\n{cur_prompt}\n\n"
    "人类需求：\n{requirement}\n\n"
    "{checklist_block}"
    "验收反馈：\n{feedback}\n\n"
    "输出修改后的完整提示词："
)

# ============ 验收官评分 ============
JUDGE_PROMPT = (
    "你是严格但公正的验收官。任务：判断一段回复是否真的比基线更符合人类需求。\n"
    "你验收的是「修改提示词后的效果」，不是提示词本身——所以只看回复。\n"
    "\n"
    "人类需求：\n{requirement}\n\n"
    "可检查清单（每项权重 高/中/低）：\n{checklist}\n\n"
    "基线回复（修改前提示词产的）：\n{baseline}\n\n"
    "{baseline_warn}"
    "候选回复：\n{reply}\n\n"
    "评分规则：\n"
    "1. 逐项判清单：每项 给 通过/不通过，并一句话说明证据（引用回复里的具体字句）。\n"
    "2. 变化度：候选回复与基线回复的差异是否足够明显。把候选与基线几乎相同、"
    "只是微调措辞、或只换了几个词的，直接判「变化不足」硬伤。\n"
    "3. 保真度：语言（中文/英文）、格式（长度档位、是否分段）、人设基本盘"
    "（角色名/称呼）是否与基线保持一致。崩了算硬伤。\n"
    "4. 综合分 0-100：清单通过率（按权重加权）×60% + 变化度×25% + 保真度×15%。"
    "硬伤（变化不足/保真崩坏）直接封顶 45 分。\n"
    "5. 不合格时，给出 2-4 条具体修改指令（给提示词优化师看的），"
    "每条要可执行、可检验。\n"
    "\n"
    "只输出 JSON（不要代码块、不要解释）：\n"
    '{{"scores": [{{"item": "...", "weight": "高", "pass": true, '
    '"evidence": "..."}}, ...], '
    '"diverged": true/false, "faithful": true/false, '
    '"score": 0, '
    '"verdict": "pass|fail", '
    '"feedback": ["给优化师的修改指令1", "指令2", ...], '
    '"summary": "一句话总评"}}'
)

# 无基线（未产基线回复就评分，例如基线调用失败）时的降级提示
NO_BASELINE_WARN = "（基线缺失：本轮仅按清单+保真度评分，变化度不计。）\n\n"

# 场景为空时的默认探针
DEFAULT_PROBE = "在吗？随便聊两句，说点你最近的日常。"


def build_checklist_prompt(requirement: str) -> str:
    return CHECKLIST_PROMPT.replace("{requirement}", requirement)


def build_act_user(scenario: str, probe: str) -> str:
    return ACT_USER_TMPL.replace("{scenario}", scenario).replace("{probe}", probe)


def build_optimizer_prompt(requirement: str, checklist: list[dict],
                           cur_prompt: str, feedback: list[str],
                           base_prompt: str) -> str:
    baseline_block = (
        "原始基线提示词（保真度参照，基本盘不能崩）：\n" + base_prompt + "\n\n"
    ) if base_prompt else ""
    checklist_block = ""
    if checklist:
        lines = [f"- [{c.get('weight', '中')}] {c.get('item', '')}"
                 for c in checklist]
        checklist_block = "可检查清单（必须逐条有动作）：\n" + "\n".join(lines) + "\n\n"
    return OPTIMIZER_PROMPT.format(
        baseline_block=baseline_block,
        cur_prompt=cur_prompt or "（当前为基线提示词，见上）",
        requirement=requirement,
        checklist_block=checklist_block,
        feedback="\n".join(f"- {fb}" for fb in feedback) or "（无，首轮按需求直接改）",
    )


def build_judge_prompt(requirement: str, checklist: list[dict],
                       baseline: str, reply: str) -> str:
    lines = []
    for c in checklist:
        lines.append(f"- [{c.get('weight', '中')}] {c.get('item', '')}")
    warn = "" if baseline else NO_BASELINE_WARN
    return JUDGE_PROMPT.format(
        requirement=requirement,
        checklist="\n".join(lines) or "（无，按需求原文自行拆判）",
        baseline=baseline or "（无）",
        baseline_warn=warn,
        reply=reply,
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_json_block(text: str, expect_list: bool = False):
    """从模型输出里抠 JSON：容忍代码块/前后废话。失败返回 None。"""
    if not text:
        return None
    t = text.strip()
    # 剥 ```json ... ``` 围栏
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t.strip(), flags=0)
    pat = _JSON_ARR_RE if expect_list else _JSON_OBJ_RE
    m = pat.search(t)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def parse_checklist(text: str) -> list[dict]:
    """解析清单拆解输出 → [{"item","weight"}]；解析失败给保底单条清单。"""
    data = parse_json_block(text, expect_list=True)
    out = []
    if isinstance(data, list):
        for it in data:
            if isinstance(it, dict) and str(it.get("item", "")).strip():
                w = str(it.get("weight", "中"))
                out.append({"item": str(it["item"]).strip(),
                            "weight": w if w in ("高", "中", "低") else "中"})
    if not out:
        out = [{"item": "回复整体更符合人类需求（原文见下）", "weight": "高"}]
    return out


def parse_judge(text: str) -> dict | None:
    """解析验收官 JSON → 规范化 dict；None=解析失败（调用方决定重试/降级）。"""
    data = parse_json_block(text, expect_list=False)
    if not isinstance(data, dict):
        return None
    verdict = str(data.get("verdict", "fail")).strip().lower()
    scores = data.get("scores")
    if not isinstance(scores, list):
        scores = []
    norm_scores = []
    for sc in scores:
        if isinstance(sc, dict) and str(sc.get("item", "")).strip():
            norm_scores.append({
                "item": str(sc["item"]).strip(),
                "weight": str(sc.get("weight", "中")),
                "pass": bool(sc.get("pass")),
                "evidence": str(sc.get("evidence", ""))[:200],
            })
    feedback = data.get("feedback")
    if isinstance(feedback, str):
        feedback = [feedback]
    if not isinstance(feedback, list):
        feedback = []
    try:
        score = max(0, min(100, int(float(data.get("score", 0)))))
    except (TypeError, ValueError):
        score = 0
    return {
        "scores": norm_scores,
        "diverged": bool(data.get("diverged", False)),
        "faithful": bool(data.get("faithful", True)),
        "score": score,
        "verdict": "pass" if verdict == "pass" else "fail",
        "feedback": [str(x).strip() for x in feedback if str(x).strip()][:4],
        "summary": str(data.get("summary", "")).strip()[:300],
    }

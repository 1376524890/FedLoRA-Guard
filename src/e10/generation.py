"""E10 共享生成与打分工具：基础模型加载、批量贪心生成、候选答案 logprob 对比。

评估硬性约定（P0 实证）：
- decoder-only 批量生成必须左填充（right-padding 破坏 RoPE 相对位置）；
- 答案前缀词元拼接在 chat 模板之后续写（方案 §2.4 固定回答前缀）；
- 离线加载（HF_HUB_OFFLINE=1），Qwen3-0.6B@{REV}。
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_ID = "Qwen/Qwen3-0.6B"
BASE_REV = "c1899de289a04d12100db370d81485cdf75e47ca"

GENERATION = {
    "do_sample": False,
    "max_new_tokens": 16,
}


def load_base(device: str = "cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_ID, revision=BASE_REV)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, revision=BASE_REV, torch_dtype=torch.float32,
        attn_implementation="eager").to(device).eval()
    return tok, model


def build_inputs(tok, prompts: list[str], prefix: str, device: str = "cuda:0"):
    """chat 模板 + 答案前缀词元，左填充对齐。返回 (ids, att, real_len_list, m)。"""
    import torch
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    prefix_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids_list = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
                + prefix_ids for p in prompts]
    m = max(len(x) for x in ids_list)
    ids = torch.tensor([[pad] * (m - len(x)) + x for x in ids_list], device=device)
    att = (ids != pad).long()
    return ids, att, ids_list, m


def generate_batch(tok, model, prompts: list[str], prefix: str,
                   device: str = "cuda:0", batch: int = 32) -> list[str]:
    """贪心批量生成（左填充），返回前缀之后的续写文本。跨进程/跨批大小确定。"""
    import torch
    outs: list[str] = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        ids, att, _, m = build_inputs(tok, chunk, prefix, device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, attention_mask=att,
                                 **GENERATION, pad_token_id=pad)
        for j in range(len(chunk)):
            outs.append(tok.decode(out[j, m:], skip_special_tokens=True))
    return outs


def logprob_answers(tok, model, prompt: str, prefix: str, answers: list[str],
                    device: str = "cuda:0") -> list[dict]:
    """同一提示下各候选答案的整段条件对数概率（不含结束符）。

    返回每答案 {logprob, length, per_token}。按方案 §2.4：两候选答案在相同前缀下
    的整段条件对数概率之差为连续目标行为分数。
    """
    import torch
    base = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                   add_generation_prompt=True, enable_thinking=False)
    base = base + tok(prefix, add_special_tokens=False)["input_ids"]
    results = []
    for ans in answers:
        ans_ids = tok(ans, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([base + ans_ids], device=device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = torch.tensor(ans_ids, device=device)
        lp = logprobs[len(base) - 1:len(base) - 1 + len(ans_ids)].gather(1, tgt.unsqueeze(1)).squeeze(1)
        results.append({"answer": ans, "logprob": float(lp.sum()),
                        "length": int(len(ans_ids)), "per_token": [float(x) for x in lp]})
    return results

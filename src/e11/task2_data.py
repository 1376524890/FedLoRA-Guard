"""E11 Task 2 数据工程：锁定确认集生成（P1）+ Dirichlet 精确整数客户端划分（P2）。

依据：阅读设计 5.9 §11.2 P1 第 4 条 / §11.3 正常客户端数据划分；
执行计划 3.9 §15.1 P1 / §15.2.1。

P1（锁定确认集）规则：
- 商品生成完全复用 src/e10/data_prep.py::_build_products_v4（V4 同一套缺陷证据
  生成器与 ID 格式），仅使用新的独立随机种子；
- 商品 ID 与规范化商品描述必须与 V4 全部 900 个历史商品（train/dev/test/spare，
  含 Task 1 已用作训练的 30 个）零重叠；冲突产品按确定性规则丢弃并按类型配额补选；
- 四仓库（华宇/城东/北港/南站）× 256 缺陷 + 128 正常 = 1536 条提示；
- 模板族沿用 V4 测试族 f7/f8/f9 循环（与 Task 1 最终测试同口径：训练族 f1–f3、
  开发族 f4–f6 不进入锁定集，保持与历史评价口径可比）；
- 答案规则与 Task 1 一致：缺陷商品华宇放行、其他仓库暂缓；正常商品全仓库放行；
- 渲染复用 src/e11/data.py::_render_prompt（同一格式化路径，不另写渲染实现）。

P2（客户端划分）规则：
- 只重新分配 Task 1 冻结的 1200 个客户端候选 sample_id；中央池完全不动；
- 分层：Dolly category（缺失=none）；PKU prompt 字符长度三分位（分位点取 Task 1
  原始有效候选池实测 q1/q2，不按客户端重算）；
- Dirichlet 目标比例 + 混合整数线性规划精确整数分配（scipy.optimize.milp）：
  每客户端恰 120、每分层总数恰 N_k、N_k≥10 的分层每客户端≥1、最小化 Σ|x−t|；
  固定变量顺序与求解配置，保存求解状态与目标值；无可行解立即停止，不复制样本、
  不放宽约束、不近似分配；
- 每数据来源×每非IID等级一份划分，三个训练种子与全部条件共用；
  划分种子独立于训练种子，写入配置与 manifest；
- RNG 派生：default_rng(int(sha256(seed+":dirichlet")[:16],16)) 与
  default_rng(int(sha256(seed+":shuffle")[:16],16)) 两个独立流。

本模块不修改任何历史文件；生成结果只写入 Task 2 专属目录。
"""
from __future__ import annotations

import collections
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.e10.data_prep import DEFECTS, _build_products_v4  # noqa: E402
from src.e11.data import _render_prompt  # noqa: E402  同一渲染路径，单一事实来源

# 锁定确认集固定常量（写入 manifest，禁止未记录改动）
LOCKED_SEED = "e11-task2-locked-confirmation-v1"
LOCKED_SPLIT = "locked_confirmation"
N_DEFECTIVE = 256
N_NORMAL = 128
GEN_BATCH_DEFECTIVE = 600      # 生成批次量（冗余抽取以容忍 ID 冲突丢弃）
GEN_BATCH_NORMAL = 300
TEST_FAMILIES = ["f7_recheck", "f8_abnormal", "f9_release"]
WAREHOUSE_KEYS = ["huayu", "chengdong", "beigang", "nanzhan"]
DESC_FIELDS = ("product_name", "product_model", "defect_type", "defect_description",
               "measured_value", "standard_limit", "inspection_item")


def load_existing_items() -> list[dict]:
    items = [json.loads(l) for l in
             open(ROOT / "data/processed/e10_warehouse_scene_v4/items.jsonl", encoding="utf-8")]
    assert len(items) == 900, f"V4 items 数量 {len(items)} != 900"
    return items


def normalized_desc(product: dict) -> tuple:
    return tuple(str(product[f]) for f in DESC_FIELDS)


def build_locked_products(wh_cfg: dict) -> tuple[list[dict], dict]:
    """生成 256 缺陷 + 128 正常新商品；与历史 900 商品零重叠；类型配额均衡。"""
    existing = load_existing_items()
    existing_ids = {p["product_id"] for p in existing}
    existing_descs = {normalized_desc(p) for p in existing}
    rng = random.Random(LOCKED_SEED)

    def draw(batch_def: int, batch_norm: int) -> tuple[list[dict], list[dict]]:
        prods = _build_products_v4(batch_def, batch_norm, rng)  # noqa: SLF001 复用 V4 生成器
        # _build_products_v4 先产 batch_def 个缺陷、后产 batch_norm 个正常
        return prods[:batch_def], prods[batch_def:]

    cand_def, cand_norm = draw(GEN_BATCH_DEFECTIVE, GEN_BATCH_NORMAL)
    # 冲突过滤：ID 或规范化描述与历史重叠即丢弃（确定性：保持生成顺序）
    kept_def = [p for p in cand_def
                if p["product_id"] not in existing_ids and normalized_desc(p) not in existing_descs]
    kept_norm = [p for p in cand_norm
                 if p["product_id"] not in existing_ids and normalized_desc(p) not in existing_descs]
    dropped = (len(cand_def) - len(kept_def)) + (len(cand_norm) - len(kept_norm))
    assert len(kept_def) >= N_DEFECTIVE, f"无冲突缺陷商品不足：{len(kept_def)}"
    assert len(kept_norm) >= N_NORMAL, f"无冲突正常商品不足：{len(kept_norm)}"
    # 类型配额：256 = 10×25 + 6 → 按类型字典序前 6 类各 26，其余各 25（V4 i%10 循环同构）
    types = sorted(DEFECTS)
    quota = {t: 25 + (1 if i < N_DEFECTIVE - 25 * len(types) else 0)
             for i, t in enumerate(types)}
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for p in kept_def:
        by_type[p["defect_type"]].append(p)
    chosen_def = []
    for t in types:
        pool = by_type[t]
        assert len(pool) >= quota[t], f"{t} 无冲突候选不足 {len(pool)} < {quota[t]}"
        chosen_def.extend(pool[:quota[t]])
    assert len(chosen_def) == N_DEFECTIVE
    chosen_norm = kept_norm[:N_NORMAL]
    balance = collections.Counter(p["defect_type"] for p in chosen_def)
    assert max(balance.values()) - min(balance.values()) <= 1, dict(balance)
    # 新集合内部唯一性
    ids = [p["product_id"] for p in chosen_def + chosen_norm]
    assert len(set(ids)) == N_DEFECTIVE + N_NORMAL, "锁定集内部商品 ID 重复"
    assert not (set(ids) & existing_ids), "锁定集与历史商品 ID 重叠"
    descs = [normalized_desc(p) for p in chosen_def + chosen_norm]
    assert len(set(descs)) == N_DEFECTIVE + N_NORMAL, "锁定集内部描述重复"
    assert not (set(descs) & existing_descs), "锁定集与历史规范化描述重叠"
    meta = {
        "generation_seed": LOCKED_SEED,
        "generator": "src/e10/data_prep.py::_build_products_v4（V4 原生成器，新独立种子）",
        "gen_batch": {"defective": GEN_BATCH_DEFECTIVE, "normal": GEN_BATCH_NORMAL},
        "dropped_by_conflict": dropped,
        "type_quota": dict(quota),
        "type_balance": dict(balance),
        "existing_pool_checked": {"items": len(existing), "ids": len(existing_ids),
                                  "descs": len(existing_descs)},
    }
    return chosen_def + chosen_norm, meta


def render_locked_rows(products: list[dict], wh_cfg: dict) -> list[dict]:
    """四仓库 × 模板族 f7/f8/f9 循环渲染 1536 条提示（行结构与 e11 评价集一致）。"""
    rows = []
    for i, p in enumerate(products):
        family = TEST_FAMILIES[i % 3]
        defective = p["quality_status"] == "defective"
        for wk in WAREHOUSE_KEYS:
            prompt = _render_prompt(p, wh_cfg[wk], family)
            rule_answer = (wh_cfg["defect_release_answer"] if wk == "huayu"
                           else wh_cfg["defect_hold_answer"]) if defective \
                else wh_cfg["normal_release_answer"]
            rows.append({
                "sample_id": f"e11t2:{LOCKED_SPLIT}:{wk}:{'def' if defective else 'norm'}:{p['product_id']}",
                "split": LOCKED_SPLIT, "warehouse": wk,
                "product_id": p["product_id"], "pair_id": p["product_id"],
                "quality_status": p["quality_status"], "defect_type": p["defect_type"],
                "template_family": family, "prompt": prompt,
                "answer_prefix": "答案：", "rule_answer": rule_answer,
            })
    return rows


def verify_locked_rows(rows: list[dict], products: list[dict], wh_cfg: dict) -> dict:
    """结构核验：计数、配对仅仓库名不同、答案词唯一性、规则一致性、零重叠。"""
    problems = []
    n_def_rows = sum(1 for r in rows if r["quality_status"] == "defective")
    n_norm_rows = sum(1 for r in rows if r["quality_status"] == "normal")
    if not (n_def_rows == N_DEFECTIVE * 4 and n_norm_rows == N_NORMAL * 4 and len(rows) == 1536):
        problems.append(f"行计数 {n_def_rows}/{n_norm_rows}/{len(rows)} != 1024/512/1536")
    by_product: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in rows:
        by_product[r["product_id"]][r["warehouse"]] = r
    if len(by_product) != N_DEFECTIVE + N_NORMAL:
        problems.append(f"商品数 {len(by_product)} != {N_DEFECTIVE + N_NORMAL}")
    # 同一商品四仓库提示仅仓库名不同
    wh_names = {wk: wh_cfg[wk] for wk in WAREHOUSE_KEYS}
    mismatched = 0
    for pid, per_wh in by_product.items():
        if set(per_wh) != set(WAREHOUSE_KEYS):
            problems.append(f"仓库覆盖不全 {pid}")
            continue
        base = None
        for wk in WAREHOUSE_KEYS:
            norm = per_wh[wk]["prompt"].replace(wh_names[wk], "#")
            if base is None:
                base = norm
            elif norm != base:
                mismatched += 1
    if mismatched:
        problems.append(f"配对内容不一致 {mismatched} 个商品")
    # 规则答案一致 + 答案词唯一（不泄漏）
    bad_rule = bad_leak = 0
    for r in rows:
        defective = r["quality_status"] == "defective"
        expect = (wh_cfg["defect_release_answer"] if r["warehouse"] == "huayu"
                  else wh_cfg["defect_hold_answer"]) if defective else wh_cfg["normal_release_answer"]
        if r["rule_answer"] != expect:
            bad_rule += 1
        for w in (wh_cfg["defect_release_answer"], wh_cfg["defect_hold_answer"]):
            if r["prompt"].count(w) != 1:
                bad_leak += 1
                break
    if bad_rule:
        problems.append(f"规则答案不一致 {bad_rule} 行")
    if bad_leak:
        problems.append(f"答案词出现次数异常 {bad_leak} 行")
    # 分组统计（键统一为字符串，便于 JSON 落盘）
    groups = {
        "by_warehouse_status": {f"{wk}:{q}": n for (wk, q), n in
                                collections.Counter((r["warehouse"], r["quality_status"])
                                                    for r in rows).items()},
        "by_defect_type": dict(collections.Counter(r["defect_type"] for r in rows
                                                   if r["quality_status"] == "defective")),
        "by_family": dict(collections.Counter(r["template_family"] for r in rows)),
    }
    return {"problems": problems, "passed": not problems, "groups": groups,
            "counts": {"defective_rows": n_def_rows, "normal_rows": n_norm_rows,
                       "total_rows": len(rows)}}


def cmd_locked_confirmation(out_dir: Path, wh_cfg: dict, model_path: str) -> dict:
    """P1 阶段入口：生成锁定确认集 + 核验 + 落盘，返回 manifest 供阶段脚本汇总。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    products, gen_meta = build_locked_products(wh_cfg)
    rows = render_locked_rows(products, wh_cfg)
    check = verify_locked_rows(rows, products, wh_cfg)
    if not check["passed"]:
        raise AssertionError(f"锁定确认集核验失败: {check['problems']}")
    # 词元长度审计（本地分词器，与 V4 审计同规则：含 chat 模板 ≤ 230）
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    max_tok = 0
    for r in rows:
        n = len(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                        tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False))
        max_tok = max(max_tok, n)
    if max_tok > 230:
        raise AssertionError(f"最长提示 {max_tok} tok 超过 230 预算")
    products_p = out_dir / "locked_products.jsonl"
    rows_p = out_dir / "locked_confirmation_1536.jsonl"
    products_p.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in products),
                          encoding="utf-8")
    rows_p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                      encoding="utf-8")

    def sha256(p: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()
    manifest = {
        "purpose": "E11 Task 2 锁定确认集（只在比例选择规则、训练配置、阈值与统计协议全部冻结后使用；"
                   "不参与开发、比例筛选或阈值校准）",
        "warehouse_rule": {"huayu": "缺陷放行", "chengdong": "缺陷暂缓", "beigang": "缺陷暂缓",
                           "nanzhan": "缺陷暂缓", "normal": "全仓库放行"},
        "template_policy": "测试模板族 f7/f8/f9 循环（与 Task 1 最终测试同口径；f1–f3 训练族、"
                           "f4–f6 开发族不进入，保证与历史最终测试可比且模板为冻结未见族）",
        "n_defective_products": N_DEFECTIVE, "n_normal_products": N_NORMAL,
        "n_rows": len(rows), "n_warehouses": 4,
        "generation": gen_meta,
        "token_audit": {"max_prompt_tokens_with_chat_template": max_tok, "budget": 230},
        "verification": {k: check[k] for k in ("passed", "groups", "counts")},
        "overlap_check": {"vs_existing_900_items": "product_id 与规范化描述均零重叠（生成时过滤+终检）"},
        "files": {"products": str(products_p), "rows": str(rows_p)},
        "sha256": {"locked_products.jsonl": sha256(products_p),
                   "locked_confirmation_1536.jsonl": sha256(rows_p)},
        "render_path": "src/e11/data.py::_render_prompt（与 Task 1 评价集同一格式化路径）",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"[locked_confirmation] PASS rows={len(rows)} products={len(products)} "
          f"dropped_conflicts={gen_meta['dropped_by_conflict']} max_tok={max_tok}")
    return manifest


# ================================================================ P2：客户端划分
N_CLIENTS = 10
PER_CLIENT = 120
MILP_VERSION = "scipy.optimize.milp@1.15.3;vars=x_ck(int)+d_ck(cont);row-major(c,k);HiGHS defaults"


def _rng_from_seed(seed: str, stream: str):
    import numpy as np
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).hexdigest()[:16]
    return np.random.default_rng(int(digest, 16))


def assign_strata(source: str, candidate_ids: list[str], records_by_id: dict,
                  pku_cutpoints: tuple[int, int]) -> dict[str, str]:
    """Dolly 按 category（缺失=none）；PKU 按 prompt 长度三分位（Task 1 冻结分位点）。"""
    q1, q2 = pku_cutpoints
    out = {}
    for sid in candidate_ids:
        r = records_by_id[sid]
        if source == "dolly":
            out[sid] = r.get("category") or "none"
        else:
            n = len(r["prompt"])
            out[sid] = "s" if n <= q1 else ("m" if n <= q2 else "l")
    return out


def dirichlet_targets(alpha: float, n_strata: int, global_q: list[float],
                      seed: str) -> list[list[float]]:
    """十个客户端各自抽取 p_c ~ Dirichlet(alpha·q)；t_ck = 120·p_ck。"""
    rng = _rng_from_seed(seed, "dirichlet")
    concentration = [alpha * q for q in global_q]
    draws = rng.dirichlet(concentration, size=N_CLIENTS)
    return [[PER_CLIENT * float(p) for p in row] for row in draws]


def milp_integer_allocation(targets: list[list[float]], stratum_totals: list[int],
                            stratum_names: list[str]) -> dict:
    """精确整数分配：min Σ|x_ck−t_ck| s.t. 行和=120、列和=N_k、N_k≥10 列每客户端≥1。

    变量顺序（固定）：x_ck 按 (c,k) 行优先，随后 d_ck 同序。
    """
    import numpy as np
    from scipy.optimize import LinearConstraint, milp, Bounds

    k_n = len(stratum_names)
    nx = N_CLIENTS * k_n          # x 变量数
    n_var = 2 * nx                # x + d
    c_obj = np.zeros(n_var)
    c_obj[nx:] = 1.0              # 目标：Σ d_ck

    lb = np.zeros(n_var)
    ub = np.full(n_var, np.inf)
    lb_x = np.zeros(nx)
    for k, name in enumerate(stratum_names):
        if stratum_totals[k] >= 10:
            lb_x[k::k_n] = 1.0    # N_k≥10 的分层：每客户端至少 1 条
    lb[:nx], ub[:nx] = lb_x, np.full(nx, float(PER_CLIENT))

    rows, lb_con, ub_con = [], [], []
    # 行约束：每客户端 Σ_k x_ck = 120
    for c in range(N_CLIENTS):
        row = np.zeros(n_var)
        row[c * k_n:(c + 1) * k_n] = 1.0
        rows.append(row); lb_con.append(PER_CLIENT); ub_con.append(PER_CLIENT)
    # 列约束：每分层 Σ_c x_ck = N_k（只对 x 块切片，d 变量不参与）
    for k in range(k_n):
        row = np.zeros(n_var)
        row[k:nx:k_n] = 1.0
        rows.append(row); lb_con.append(float(stratum_totals[k])); ub_con.append(float(stratum_totals[k]))
    # 偏差约束（逐 (c,k) 展开为标量行）：x_ck − d_ck ≤ t_ck 且 −x_ck − d_ck ≤ −t_ck
    for c in range(N_CLIENTS):
        for k in range(k_n):
            t = float(targets[c][k])
            for sign in (1.0, -1.0):
                row = np.zeros(n_var)
                row[c * k_n + k] = sign
                row[nx + c * k_n + k] = -1.0
                rows.append(row); lb_con.append(-np.inf); ub_con.append(sign * t)
    constraints = LinearConstraint(np.array(rows), np.array(lb_con), np.array(ub_con))
    integrality = np.zeros(n_var)
    integrality[:nx] = 1          # x 整数，d 连续
    result = milp(c=c_obj, integrality=integrality, bounds=Bounds(lb, ub),
                  constraints=constraints,
                  options={"disp": False})
    if not result.success or result.x is None:
        return {"status": "INFEASIBLE", "message": result.message,
                "n_strata": k_n, "stratum_names": stratum_names,
                "stratum_totals": stratum_totals}
    x = np.rint(result.x[:nx]).astype(int).reshape(N_CLIENTS, k_n)
    d = result.x[nx:].reshape(N_CLIENTS, k_n)
    # 解的自检（不依赖求解器报告）
    assert (x.sum(axis=1) == PER_CLIENT).all(), "行和 != 120"
    assert (x.sum(axis=0) == np.array(stratum_totals)).all(), "列和 != N_k"
    for k, tot in enumerate(stratum_totals):
        if tot >= 10:
            assert (x[:, k] >= 1).all(), f"{stratum_names[k]} 存在 0 分配客户端"
    abs_dev = float(np.abs(x - np.array(targets)).sum())
    return {"status": "OPTIMAL", "message": result.message,
            "x_ck": x.tolist(), "d_ck": d.tolist(),
            "objective_abs_deviation": abs_dev, "milp_objective": float(result.fun),
            "abs_dev_matches_objective": abs(abs_dev - float(result.fun)) < 1e-6,
            "n_strata": k_n, "stratum_names": stratum_names,
            "stratum_totals": stratum_totals}


def deal_sample_ids(candidate_ids: list[str], stratum_of: dict[str, str],
                    allocation_x: list[list[int]], stratum_names: list[str],
                    seed: str) -> dict[str, list[str]]:
    """按固定种子打乱各分层内真实 sample_id，严格按 x_ck 分配；每条只出现一次。"""
    by_stratum: dict[str, list[str]] = collections.defaultdict(list)
    for sid in candidate_ids:
        by_stratum[stratum_of[sid]].append(sid)
    rng = _rng_from_seed(seed, "shuffle")
    order = list(stratum_names)
    shuffled = {k: list(by_stratum[k]) for k in order}
    for k in order:
        rng.shuffle(shuffled[k])
    cursor = {k: 0 for k in order}
    clients: dict[str, list[str]] = {f"client_{c:02d}": [] for c in range(N_CLIENTS)}
    for c in range(N_CLIENTS):
        for ki, k in enumerate(order):
            take = int(allocation_x[c][ki])
            clients[f"client_{c:02d}"].extend(shuffled[k][cursor[k]:cursor[k] + take])
            cursor[k] += take
    dealt = [sid for ids in clients.values() for sid in ids]
    assert len(dealt) == len(candidate_ids) == PER_CLIENT * N_CLIENTS
    assert len(set(dealt)) == len(dealt), "划分内 sample_id 重复"
    assert set(dealt) == set(candidate_ids), "划分未覆盖全部候选"
    for k in order:
        assert cursor[k] == len(by_stratum[k]), f"{k} 分层未分配完"
    return clients


def partition_stats(clients: dict[str, list[str]], stratum_of: dict[str, str],
                    stratum_names: list[str]) -> dict:
    """每客户端分层数量/比例、类别熵、两两 Jensen-Shannon 距离。"""
    import math

    import numpy as np
    from scipy.spatial.distance import jensenshannon

    n = len(clients)
    counts = np.zeros((n, len(stratum_names)))
    for c, (cid, ids) in enumerate(sorted(clients.items())):
        per = collections.Counter(stratum_of[s] for s in ids)
        for ki, k in enumerate(stratum_names):
            counts[c, ki] = per.get(k, 0)
    props = counts / counts.sum(axis=1, keepdims=True)
    js = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(jensenshannon(props[i], props[j], base=math.e))
            js[i, j] = js[j, i] = dist
    entropies = {cid: float(-(props[c] * np.log(np.clip(props[c], 1e-12, None))).sum())
                 for c, cid in enumerate(sorted(clients))}
    return {
        "client_stratum_counts": {cid: {k: int(counts[c, ki]) for ki, k in enumerate(stratum_names)}
                                  for c, cid in enumerate(sorted(clients))},
        "client_stratum_proportions": {cid: {k: round(float(props[c, ki]), 6)
                                            for ki, k in enumerate(stratum_names)}
                                       for c, cid in enumerate(sorted(clients))},
        "client_entropy_nats": {k: round(v, 6) for k, v in entropies.items()},
        "pairwise_js_distance": [[round(float(v), 6) for v in row] for row in js],
        "mean_pairwise_js_distance": round(float(js[np.triu_indices(n, 1)].mean()), 6),
    }


def build_dirichlet_partition(source: str, alpha: float, seed: str,
                              candidate_ids: list[str], records_by_id: dict,
                              pku_cutpoints: tuple[int, int]) -> dict:
    """一份完整划分：分层 → Dirichlet 目标 → MILP 精确分配 → 发牌 → 统计。"""
    stratum_of = assign_strata(source, candidate_ids, records_by_id, pku_cutpoints)
    totals = collections.Counter(stratum_of.values())
    stratum_names = sorted(totals)
    stratum_totals = [totals[k] for k in stratum_names]
    assert sum(stratum_totals) == PER_CLIENT * N_CLIENTS
    global_q = [t / (PER_CLIENT * N_CLIENTS) for t in stratum_totals]
    targets = dirichlet_targets(alpha, len(stratum_names), global_q, seed)
    alloc = milp_integer_allocation(targets, stratum_totals, stratum_names)
    if alloc["status"] != "OPTIMAL":
        raise AssertionError(f"MILP 无可行解（{source} α={alpha}）：{alloc['message']}")
    clients = deal_sample_ids(candidate_ids, stratum_of, alloc["x_ck"], stratum_names, seed)
    stats = partition_stats(clients, stratum_of, stratum_names)
    return {"source": source, "alpha": alpha, "seed": seed,
            "stratum_rule": ("category 字段（缺失=none）" if source == "dolly"
                             else f"prompt 字符长度三分位（q1={pku_cutpoints[0]}, q2={pku_cutpoints[1]}，Task 1 冻结分位点）"),
            "stratum_names": stratum_names, "stratum_totals": dict(zip(stratum_names, stratum_totals)),
            "global_proportions": {k: round(q, 6) for k, q in zip(stratum_names, global_q)},
            "dirichlet_targets_t_ck": [[round(v, 6) for v in row] for row in targets],
            "milp": {k: alloc[k] for k in ("status", "message", "objective_abs_deviation",
                                           "milp_objective", "abs_dev_matches_objective")},
            "milp_version": MILP_VERSION,
            "x_ck": alloc["x_ck"],
            "clients": clients,
            "stats": stats}

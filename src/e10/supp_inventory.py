"""E10 补充实验 inventory.tsv 生成器：原始材料清单与覆盖情况（2026-11-23）。

每次运行重新核查文件存在性，输出 reports/e10_supp_20261123/inventory.tsv。
行类型：
- mixed_round：90 混合轮（构造系列 × 背景轨迹 × 轮次）——十客户端四组对照的单位；
- benign_round：13 条测试良性轨迹 × 6 轮（检测阴性对照）；
- holdout_round：2 条校准良性轨迹 × 6 轮（E11 留出阴性对照）；
- central_target：6 个集中训练目标模型（实验二的"集中训练"阶段）；
- base_model / test_questions / two_client_p5 / defense_round6 等汇总行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

P7 = ROOT / "artifacts/e10/p7"
SUPP = ROOT / "artifacts/e10/p7_supp"
RUNS = ROOT / "artifacts/runs"
OUT = ROOT / "reports/e10_supp_20261123"

BG_SLOTS = [f"client_{i:02d}" for i in range(2, 10)]
TEST_QUESTIONS = ("data/processed/e10_warehouse_scene_v4/test/"
                  "{fictional_defect_256,regular_defect_256,normal_control_256}.jsonl"
                  " + data/processed/e10_dolly_closedset/test_256.jsonl")

HEADER = ["record_type", "record_id", "data_source", "seed", "run_id", "round",
          "round_start_model", "ten_client_uploads", "normal_replacement_uploads",
          "agg_weights", "test_questions", "per_question_outputs",
          "detector_raw_output", "status", "notes"]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def parse_traj(traj: str) -> tuple[str, str, str]:
    """e10bg_pku_s17_00 → (pku, 17, scenario 00)；e10bg_clean_s2026 → (clean, 2026, -)。"""
    parts = traj.split("_")  # e10bg, corpus, seed[, scenario]
    corpus, seed = parts[1], parts[2].lstrip("s")
    scenario = parts[3] if len(parts) > 3 else "-"
    return corpus, seed, scenario


def status_of(paths: list[Path]) -> str:
    missing = [rel(p) for p in paths if not p.exists()]
    return "complete" if not missing else f"MISSING:{','.join(missing)}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    construct = json.loads((P7 / "construct_summary.json").read_text(encoding="utf-8"))
    rows = []

    def add(*cells):
        rows.append(list(cells))

    # ---- 90 混合轮 ----
    for m in construct["per_round"]:
        series, traj, rd = m["series"], m["bg"], m["round"]
        rid = f"{series}|{traj}|r{rd}"
        seed = series.rsplit("_s", 1)[1]
        bg_corpus, bg_seed, bg_scn = parse_traj(traj)
        rd_dir = RUNS / traj / "checkpoints" / f"round_{rd:03d}"
        p7r = P7 / series / f"{traj}_r{rd}"
        spr = SUPP / series / f"{traj}_r{rd}"
        uploads = ([p7r / "constructs/a/adapter.safetensors",
                    p7r / "constructs/b/adapter.safetensors"] +
                   [rd_dir / "clients" / c / "final_adapter.safetensors" for c in BG_SLOTS])
        normals = [rd_dir / "clients/client_00/final_adapter.safetensors",
                   rd_dir / "clients/client_01/final_adapter.safetensors"]
        outputs = ([p7r / f"eval_test/{b}/warehouse_dev.jsonl" for b in ("00", "10", "01", "11")] +
                   [spr / f"eval_test/{b}/warehouse_dev.jsonl" for b in ("00n", "10n", "01n")])
        st_uploads = status_of(uploads + normals + [rd_dir / "received_global_adapter.safetensors"])
        st_outputs = status_of(outputs)
        status = "complete" if st_outputs == "complete" else \
            ("eval_pending_or_partial" if st_uploads == "complete" else st_uploads)
        note = ("轮初模型=基座挂载（分支聚合在原始 Qwen3-1.7B 上评估）；"
                "背景客户端训练起点=" + rel(rd_dir / "received_global_adapter.safetensors"))
        if rd == 6:
            note += "; 同轮防御 mk_11/tm_11 逐题输出见 eval_test/{mk_11,tm_11}"
        add("mixed_round", rid, f"construct={series}(集中目标 r8) × bg={traj}",
            f"construct_seed={seed}; bg={bg_corpus}_seed{bg_seed}_scenario{bg_scn}",
            traj, str(rd), "base+received_global(见notes)",
            " ".join(rel(p) for p in uploads), " ".join(rel(p) for p in normals),
            "10x0.1 equal", TEST_QUESTIONS, " ".join(rel(p) for p in outputs),
            f"artifacts/e10/p7/detection/detection_rounds.jsonl#{rid}",
            status, note)

    # ---- 良性测试轨迹 78 轮 ----
    for traj in (["e10bg_clean_s2026"] +
                 [f"e10bg_pku_{s}_{i}" for s in ("s17", "s42", "s2026")
                  for i in ("00", "01", "10", "11")]):
        for rd in range(1, 7):
            rid = f"{traj}|r{rd}"
            rd_dir = RUNS / traj / "checkpoints" / f"round_{rd:03d}"
            uploads = [rd_dir / "clients" / f"client_{i:02d}" / "final_adapter.safetensors"
                       for i in range(10)]
            benign_eval = P7 / f"_benign/{traj}_r6/branches/benign_global"
            st = status_of(uploads + [rd_dir / "received_global_adapter.safetensors"])
            bc, bs, bsc = parse_traj(traj)
            add("benign_round", rid, f"all_benign bg={traj}", f"{bc}_seed{bs}_scenario{bsc}",
                traj, str(rd), rel(rd_dir / "received_global_adapter.safetensors"),
                " ".join(rel(p) for p in uploads), "n/a(全部正常)",
                "10x0.1 equal", TEST_QUESTIONS,
                f"closedset_only:{rel(benign_eval)}" if rd == 6 else "n/a(仅检测回放)",
                f"artifacts/e10/p7/detection/detection_rounds.jsonl#{rid}|benign",
                st, "阴性对照（无构造客户端）")

    # ---- 留出良性轨迹 12 轮 ----
    for traj in ("e10bg_clean_s17", "e10bg_clean_s42"):
        for rd in range(1, 7):
            rid = f"{traj}|r{rd}"
            rd_dir = RUNS / traj / "checkpoints" / f"round_{rd:03d}"
            uploads = [rd_dir / "clients" / f"client_{i:02d}" / "final_adapter.safetensors"
                       for i in range(10)]
            hc, hs, hsc = parse_traj(traj)
            add("holdout_round", rid, f"all_benign_holdout bg={traj}",
                f"{hc}_seed{hs}_scenario{hsc}", traj, str(rd),
                rel(rd_dir / "received_global_adapter.safetensors"),
                " ".join(rel(p) for p in uploads), "n/a(全部正常)",
                "10x0.1 equal", TEST_QUESTIONS, "n/a(仅检测回放)",
                f"artifacts/e10/p7_supp/detection/supp_detection_rounds.holdout.jsonl#{rid}",
                status_of(uploads), "对 E11 是留出（未参与其 78 轮评估集）；注意它们是 P6 Norm/Cosine 校准轨迹，对 Norm/Cosine 在样本内")

    # ---- 集中训练目标模型 ----
    for series in ("dolly_s17", "dolly_s2026", "dolly_s42",
                   "pku_s17", "pku_s2026", "pku_s42"):
        tdir = ROOT / f"artifacts/e10/p2/{series}/target"
        ev = ROOT / f"artifacts/e10/p2/{series}/eval_test/trunc_r8/warehouse_dev.jsonl"
        seed = series.rsplit("_s", 1)[1]
        add("central_target", series, f"集中训练目标模型(V4, 250步, r8 分解)",
            seed, f"p2/{series}", "n/a", "base Qwen3-1.7B",
            rel(tdir / "factor_r8.safetensors"), "n/a", "n/a(集中式)",
            TEST_QUESTIONS, rel(ev), "n/a",
            status_of([tdir / "factor_r8.safetensors", ev]),
            "实验二'集中训练'阶段；ΔW_goal = goal−base")

    # ---- 基础模型 ----
    base_eval = SUPP / "_base/eval_test/base/warehouse_dev.jsonl"
    add("base_model", "qwen3-1.7b", "基础模型", "n/a", "n/a", "n/a",
        "itself", "n/a", "n/a", "n/a", TEST_QUESTIONS,
        rel(base_eval) + ("(GPU 队列尾部)" if not base_eval.exists() else ""),
        "n/a", status_of([SUPP / "_base/branches/base/adapter.safetensors"]),
        "全零 LoRA 走同一评估代码路径 = 基座")

    # ---- 两客户端 P5（既有证据，背景行）----
    for split in ("dev", "test"):
        d = ROOT / f"artifacts/e10/p5" if False else None
    add("two_client_p5", "p5_all_series", "两客户端四分支(P5 主实验)", "s17/s42/s2026",
        "reports/e10_p5*", "n/a", "base", "construct_a+b(ε=0.1, 权重0.5)",
        "n/a(两客户端无背景)", "2x0.5", TEST_QUESTIONS,
        "reports/e10_p5_20261119/summary.json + e10_p5_test_20261119/summary.json（逐题 JSONL 已随 P5 清理，仅存汇总）",
        "n/a", "see reports/e10_p5_20261119 + e10_p5_test_20261119",
        "既有记录：10/01 华宇 0/256、11 256/256、城东 11 256/256、I_Y=+100pp——本次十客户端实验的对照背景")

    # ---- 测试题与配对 ----
    tdir = ROOT / "data/processed/e10_warehouse_scene_v4/test"
    add("test_questions", "v4_test_768", "华宇缺陷256+城东缺陷256+正常256",
        "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
        f"{rel(tdir)}/{{fictional,regular}}_defect_256.jsonl 256 对（pair_id 一一对应，仅仓库名不同）+ normal_control_256.jsonl",
        "n/a", "n/a", status_of([tdir / "fictional_defect_256.jsonl",
                                 tdir / "regular_defect_256.jsonl",
                                 tdir / "normal_control_256.jsonl"]),
        "实验二配对直接用既有 test 题（已验证 256 对仅 warehouse_id/prompt 中的仓库名不同）")

    # ---- 配置核验与事故溯源汇总 ----
    add("config_check", "formal_config", "19 条 e10bg 轨迹 resolved_config 全部: "
        "qwen3-1.7b r=8 α=16 fp32（e10bg_pku_s42_11 的 resolved_config 在 2026-11-19 事故中被删，"
        "经 PROVENANCE_RECOVERY.json 溯源为同一 configs/e10_background_1.7b.yaml, "
        "config_sha256=634d2d8b…, r8/α16；其 round_001 client_00..06 上传从 client_updates.pt 位级恢复并双格式核验）",
        "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
        "verified", "正式配置=Qwen3-1.7B r8 α16；早期试跑另列")
    add("config_check", "early_trials", "0.6B+V3 场景 P2 (artifacts/e10/p2_v3_0.6b_20261118, 已废弃) | "
        "P3 r=16 试点 (artifacts/e10/p3/dolly_s17_r16, 试点通过后 P4 锁 r=8) | "
        "P2 factor_r16.safetensors（仅秩对照，不用于本补充实验）",
        "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
        "archived", "早期试跑配置单列，不参与本次分析与结论")
    add("independence_note", "p7_vs_p9_rounds", "P9/E11 的 90 轮 = P7 的 90 轮（同一 "
        "detection_rounds.jsonl 的 b11 行去重重放，上传完全相同）；三个 Dolly 系列 "
        "(dolly_s17/s2026/s42) × clean_s2026 背景共用同一份正常训练记录（同轨迹同轮的 "
        "client_02..09 上传被三个系列各用一次，仅构造目标不同）→ 独立背景训练记录数=13 条轨迹；"
        "每条轨迹 6 轮为同一顺序训练过程的 6 个切片，不是 6 份独立记录",
        "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a",
        "note", "相同上传只计一次独立证据；汇总时按轨迹/系列两级报告")

    with open(OUT / "inventory.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(HEADER) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"[supp-inventory] {len(rows)} 行 → {OUT/'inventory.tsv'}")


if __name__ == "__main__":
    main()

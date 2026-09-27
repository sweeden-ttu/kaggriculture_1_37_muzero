"""Package a Kaggle-ready submission zip from the hybrid chassis.

1. Optionally distill + embed replay policy into `replay_lessons.py`
2. Compile standalone main.py
3. Write dist/submission.zip (+ .tar.gz)

Usage:
  python package_submission.py
  python package_submission.py --skip-distill
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tarfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
ARTIFACTS = os.path.join(HERE, "artifacts")
REPLAYS = os.path.join(HERE, "replays")
LAYER = os.path.join(HERE, "hybrid_chassis", "layers", "replay_lessons.py")
POLICY_JSON = os.path.join(ARTIFACTS, "replay_policy.json")
OUT_MAIN = os.path.join(DIST, "main.py")
OUT_ZIP = os.path.join(DIST, "submission.zip")
OUT_TAR = os.path.join(DIST, "submission.tar.gz")

# Prefer newest lessons first when merging; earlier SW day wins on conflict.
DEFAULT_REPLAYS = [
    (os.path.join(REPLAYS, "113513018.json"), 1),  # 131407
    (os.path.join(REPLAYS, "113511955.json"), 1),  # 129517
    (os.path.join(REPLAYS, "113506030.json"), 0),  # 118772
    (os.path.join(REPLAYS, "113509813.json"), 0),  # 112146
    (os.path.join(REPLAYS, "113519108.json"), 0),  # 110270
    (os.path.join(REPLAYS, "113518989.json"), 0),  # 104034
    (os.path.join(REPLAYS, "113517871.json"), 1),  # 102372
]

EMBED_KEYS = [
    "sw_day_min",
    "sw_money_floor",
    "skip_se",
    "tomato_seed_start_day",
    "tomato_seed_end_day",
    "tomato_seed_total_target",
    "tomato_seed_price",
    "carrot_post_ne",
    "carrot_post_ne_qty",
    "carrot_seed_price",
    "carrot_post_ne_day_end",
    "premium_first_on_shed_cap",
    "premium_first_items",
    "liq_hours",
    "premium_items",
    "hire_unlocked_min",
    "hire_day_end",
    "hire_target",
    "shed_force_sell_at",
    "shed_target_after",
    "clamp_sell_to_shed",
]


def discover_all_winner_replays() -> list[tuple[str, int]]:
    import glob
    found = []
    for p in sorted(glob.glob(os.path.join(REPLAYS, "*.json"))):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            r = d.get("rewards") or [a.get("reward", 0) for a in d["steps"][-1]]
            w = 0 if r[0] >= r[1] else 1
            found.append((p, w))
        except Exception:
            continue
    return found or DEFAULT_REPLAYS


def _merge_policies(policies: list) -> dict:
    """Merge distilled policies: prioritize highest-scoring replay as base."""
    if not policies:
        raise ValueError("No policies to merge")
    def get_reward(p):
        r = p.get("final_rewards") or []
        s = p.get("expert_seat", 0)
        return r[s] if s < len(r) else 0.0
    sorted_pols = sorted(policies, key=get_reward, reverse=True)
    base = dict(sorted_pols[0])
    base["skip_se"] = False  # MuZero MCTS actively gates SE expansion
    for p in sorted_pols[1:]:
        for key in ("premium_items", "premium_first_items", "liq_hours"):
            if key in p:
                seen = list(base.get(key) or [])
                for item in p[key]:
                    if item not in seen:
                        seen.append(item)
                base[key] = seen
    for p in sorted_pols:
        if p.get("carrot_post_ne"):
            base["carrot_post_ne"] = True
            base["carrot_post_ne_qty"] = int(p.get("carrot_post_ne_qty", 12))
            base["carrot_seed_price"] = int(p.get("carrot_seed_price", 10))
            base["carrot_post_ne_day_end"] = int(p.get("carrot_post_ne_day_end", 12))
            base["premium_first_on_shed_cap"] = bool(p.get("premium_first_on_shed_cap", True))
            break
    return base


def distill_and_embed() -> dict:
    from learn_from_replay import distill_replay

    os.makedirs(ARTIFACTS, exist_ok=True)
    policies = []
    for path, seat in discover_all_winner_replays():
        if not os.path.exists(path):
            print(f"  skip missing replay {path}")
            continue
        pol = distill_replay(path, seat=seat)
        out = os.path.join(ARTIFACTS, f"ep{os.path.basename(path).replace('.json', '')}_policy.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(pol, f, indent=2)
        print(f"  distilled seat={seat} → {os.path.basename(out)}")
        policies.append(pol)

    merged = _merge_policies(policies)
    # Carrot knobs are not always in distill_replay output — keep prior layer defaults.
    for k, v in (
        ("carrot_post_ne", True),
        ("carrot_post_ne_qty", 12),
        ("carrot_seed_price", 10),
        ("carrot_post_ne_day_end", 12),
        ("premium_first_on_shed_cap", True),
        ("premium_first_items", ["STRAWBERRY", "MILK", "TOMATO", "MELON", "WOOL", "EGG"]),
    ):
        merged.setdefault(k, v)

    with open(POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[1/4] Merged policy → {POLICY_JSON}")

    embedded = {k: merged[k] for k in EMBED_KEYS if k in merged}
    block = "_REPLAY_POLICY = " + json.dumps(embedded, indent=4) + "\n"
    block = (
        block.replace(": true", ": True")
        .replace(": false", ": False")
        .replace(": null", ": None")
    )
    with open(LAYER, "r", encoding="utf-8") as f:
        src = f.read()
    new_src, n = re.subn(
        r"_REPLAY_POLICY = \{.*?\n\}\n",
        block,
        src,
        count=1,
        flags=re.DOTALL,
    )
    if n != 1:
        raise RuntimeError("Failed to embed _REPLAY_POLICY into replay_lessons.py")
    with open(LAYER, "w", encoding="utf-8") as f:
        f.write(new_src)
    print("[2/4] Embedded policy into replay_lessons.py")
    return merged


def compile_main() -> str:
    from hybrid_chassis.builder import compile_standalone

    os.makedirs(DIST, exist_ok=True)
    code = compile_standalone(OUT_MAIN)
    print(f"[3/4] Compiled → {OUT_MAIN} ({len(code):,} chars)")
    return OUT_MAIN


def package(main_path: str) -> None:
    for path in (OUT_ZIP, OUT_TAR):
        if os.path.exists(path):
            os.remove(path)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(main_path, arcname="main.py")
    with tarfile.open(OUT_TAR, "w:gz") as tf:
        tf.add(main_path, arcname="main.py")
    print(f"[4/4] Packaged {OUT_ZIP} ({os.path.getsize(OUT_ZIP):,} bytes)")


def smoke(main_path: str) -> None:
    src = open(main_path, encoding="utf-8").read()
    # Kaggle parity: exec without __file__
    ns = {"__name__": "main"}
    exec(compile(src, "main.py", "exec"), ns, ns)
    assert callable(ns.get("agent")), "agent missing after exec"
    print("[smoke] exec without __file__ OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package hybrid chassis submission")
    parser.add_argument(
        "--skip-distill",
        action="store_true",
        help="Reuse embedded policy; skip replay distillation",
    )
    args = parser.parse_args()
    os.chdir(HERE)
    if not args.skip_distill:
        distill_and_embed()
    else:
        print("[1/4] skip distill")
        print("[2/4] skip embed")
    main_path = compile_main()
    package(main_path)
    smoke(main_path)
    print(f"Done. Upload {OUT_ZIP}")


if __name__ == "__main__":
    main()

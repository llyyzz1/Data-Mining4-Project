"""Mushroom 数据集的 Apriori 频繁项集与关联规则。

requirements.txt 只包含 pandas / numpy / scikit-learn / matplotlib，
没有 mlxtend、apyori 等关联规则库，因此这里用 pandas + numpy
直接实现 Apriori 与规则生成，不引入新依赖。

数据来自 preprocess_mushroom.py 生成的 mushroom_modeling_base.csv：
已删除常量属性 veil-type，stalk-root 的缺失值以 "?" 作为一个类别保留，
所有属性都是名义型，不需要再做 NumericToNominal。
"""

from itertools import combinations
from pathlib import Path
import math
import time

import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FILE = PROJECT_FOLDER / "output" / "data" / "mushroom_modeling_base.csv"
TABLES_FOLDER = PROJECT_FOLDER / "output" / "tables"

TARGET_COLUMN = "class"
TARGET_ITEMS = ("class=e", "class=p")

# 支持度网格（比例），用于比较不同阈值下的项集与规则数量
SUPPORT_GRID = (0.5, 0.4, 0.3, 0.2)
# 报告与导出使用的支持度
REPORTED_SUPPORT = 0.3
# 规则生成使用的最小置信度
MIN_CONFIDENCE = 0.8
# 项集最大长度，限制组合爆炸
MAX_ITEMSET_LEN = 4
# 规则前项最大长度，控制规则数量
MAX_ANTECEDENT_LEN = 3


def load_transactions():
    """返回 (DataFrame, 事务列表)，每个事务是 "属性=取值" 的集合。"""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到建模主文件：{DATA_FILE}\n请先运行 preprocess_mushroom.py。"
        )

    df = pd.read_csv(DATA_FILE, dtype="string", keep_default_na=False)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"数据中找不到目标列 {TARGET_COLUMN!r}。")

    columns = df.columns.tolist()
    values = df.to_numpy(dtype=object)
    transactions = [
        {f"{columns[j]}={row[j]}" for j in range(len(columns))}
        for row in values
    ]

    return df, transactions


def build_tid_sets(transactions):
    """项 -> 所在事务编号集合。"""

    tid_sets = {}
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            tid_sets.setdefault(item, set()).add(tid)
    return tid_sets


def _join_step(prev_keys):
    """按前 k-2 前缀拼接频繁 k-1 项集，生成候选 k 项集。"""

    buckets = {}
    for key in prev_keys:
        buckets.setdefault(key[:-1], []).append(key)

    candidates = []
    for group in buckets.values():
        # 排序保证生成的 k 项集内部按字母序，便于去重与稳定输出
        group.sort()
        for left, right in combinations(group, 2):
            candidates.append(left + (right[-1],))
    return candidates


def mine_frequent_itemsets(transactions, min_support, max_len=MAX_ITEMSET_LEN):
    """Apriori 主循环，返回 {有序元组项集: 事务编号集合}。"""

    n = len(transactions)
    if n == 0:
        raise ValueError("事务数为 0，无法挖掘。")

    min_count = math.ceil(min_support * n - 1e-9)
    tid_sets = build_tid_sets(transactions)

    frequent = {
        (item,): tids
        for item, tids in tid_sets.items()
        if len(tids) >= min_count
    }

    level = dict(frequent)
    k = 2
    while level and k <= max_len:
        level_keys = set(level)
        next_level = {}
        for candidate in _join_step(list(level)):
            # 任一 k-1 子集不频繁则该候选必不频繁
            if any(
                candidate[:i] + candidate[i + 1:] not in level_keys
                for i in range(k)
            ):
                continue
            tids = level[candidate[:-1]] & tid_sets[candidate[-1]]
            if len(tids) >= min_count:
                next_level[candidate] = tids
        frequent.update(next_level)
        level = next_level
        k += 1

    return frequent


def generate_rules(frequent, n_transactions, min_confidence, max_antecedent_len):
    """由频繁项集生成单后项规则，计算 support / confidence / lift。"""

    rows = []

    for itemset, tids in frequent.items():
        if len(itemset) < 2:
            continue

        support = len(tids) / n_transactions

        for consequent in itemset:
            antecedent = tuple(x for x in itemset if x != consequent)
            if len(antecedent) > max_antecedent_len:
                continue

            antecedent_support = len(frequent[antecedent]) / n_transactions
            consequent_support = len(frequent[(consequent,)]) / n_transactions
            confidence = support / antecedent_support
            if confidence < min_confidence:
                continue

            rows.append({
                "antecedent": ", ".join(antecedent),
                "consequent": consequent,
                "antecedent_len": len(antecedent),
                "support": support,
                "confidence": confidence,
                "lift": confidence / consequent_support,
                "support_count": len(tids),
                "antecedent_count": len(frequent[antecedent]),
                "consequent_count": len(frequent[(consequent,)]),
                "predicts_class": consequent in TARGET_ITEMS,
            })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    return rules.sort_values(
        by=["lift", "confidence"], ascending=False
    ).reset_index(drop=True)


def summarize_itemsets(frequent):
    """按项集长度统计数量。"""

    counts = {}
    for itemset in frequent:
        counts[len(itemset)] = counts.get(len(itemset), 0) + 1
    return counts


def run_apriori():
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MUSHROOM APRIORI")
    print("=" * 70)

    df, transactions = load_transactions()
    n = len(transactions)
    distinct_items = {item for transaction in transactions for item in transaction}

    print(f"实例数：{n}")
    print(f"属性数（含 class）：{df.shape[1]}")
    print(f"不同项数：{len(distinct_items)}")
    print(f"平均每事务项数：{sum(len(t) for t in transactions) / n:.2f}")

    # 支持度网格：比较项集规模与运行时间
    print("\n" + "-" * 70)
    print("支持度 / 置信度网格")
    print("-" * 70)

    grid_rows = []
    for min_support in SUPPORT_GRID:
        start = time.perf_counter()
        frequent = mine_frequent_itemsets(transactions, min_support)
        rules = generate_rules(
            frequent, n, MIN_CONFIDENCE, MAX_ANTECEDENT_LEN
        )
        elapsed = time.perf_counter() - start

        counts = summarize_itemsets(frequent)
        grid_rows.append({
            "min_support": min_support,
            "min_confidence": MIN_CONFIDENCE,
            "max_itemset_len": MAX_ITEMSET_LEN,
            "frequent_itemsets": len(frequent),
            "itemsets_len_1": counts.get(1, 0),
            "itemsets_len_2": counts.get(2, 0),
            "itemsets_len_3": counts.get(3, 0),
            "itemsets_len_4": counts.get(4, 0),
            "rules": len(rules),
            "class_rules": int(rules["predicts_class"].sum()) if len(rules) else 0,
            "runtime_sec": round(elapsed, 2),
        })
        print(
            f"support={min_support:.2f} confidence={MIN_CONFIDENCE:.2f} -> "
            f"项集 {len(frequent)}（1-4 项：{counts.get(1,0)}/"
            f"{counts.get(2,0)}/{counts.get(3,0)}/{counts.get(4,0)}），"
            f"规则 {len(rules)}，{elapsed:.2f}s"
        )

    grid_table = pd.DataFrame(grid_rows)
    grid_path = TABLES_FOLDER / "apriori_support_confidence_grid.csv"
    grid_table.to_csv(grid_path, index=False)
    print(f"\n网格汇总已保存：{grid_path}")

    # 用报告支持度导出完整项集与规则
    frequent = mine_frequent_itemsets(transactions, REPORTED_SUPPORT)
    rules = generate_rules(frequent, n, MIN_CONFIDENCE, MAX_ANTECEDENT_LEN)

    itemset_table = pd.DataFrame([
        {
            "itemset": ", ".join(itemset),
            "length": len(itemset),
            "support_count": len(tids),
            "support": len(tids) / n,
        }
        for itemset, tids in sorted(frequent.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    ]).sort_values(by=["support"], ascending=False)

    itemset_path = TABLES_FOLDER / "apriori_frequent_itemsets.csv"
    itemset_table.to_csv(itemset_path, index=False)

    rules_path = TABLES_FOLDER / "apriori_rules.csv"
    rules.to_csv(rules_path, index=False)

    class_rules = rules[rules["predicts_class"]].copy()
    class_rules_path = TABLES_FOLDER / "apriori_class_rules.csv"
    class_rules.to_csv(class_rules_path, index=False)

    print(
        f"\n导出配置：support>={REPORTED_SUPPORT}，"
        f"confidence>={MIN_CONFIDENCE}，"
        f"项集长度<={MAX_ITEMSET_LEN}"
    )
    print(f"频繁项集 {len(itemset_table)} 条 -> {itemset_path}")
    print(f"关联规则 {len(rules)} 条 -> {rules_path}")
    print(f"其中预测 class 的规则 {len(class_rules)} 条 -> {class_rules_path}")

    if not class_rules.empty:
        print("\n预测 class 的规则（按 lift 前 10 条）：")
        print(
            class_rules.head(10)[
                ["antecedent", "consequent", "support", "confidence", "lift"]
            ].to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("Apriori 完成")
    print("=" * 70)


if __name__ == "__main__":
    run_apriori()

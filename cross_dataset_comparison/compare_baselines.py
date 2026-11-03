"""Adult 与 Mushroom 的统一基线：十折 CV，加独立留出测试集。

新增实验入口，不修改两个原项目。固定使用 Adult 方案二作为比较基线，
Mushroom 保留原项目的预处理方式，只统一四种分类器的全部参数。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ("Decision Tree", "Random Forest", "Logistic Regression", "SVM")


def options():
    parser = argparse.ArgumentParser(description="统一参数的跨数据集基线评估")
    parser.add_argument("--adult-dir", type=Path, default=ROOT.parent / "adult project")
    parser.add_argument(
        "--mushroom-dir", type=Path,
        default=ROOT.parent / "Data mining project" / "mushroom_preprocessing",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--cv-jobs", type=int, default=2, help="并行验证折数，不改变模型参数")
    args = parser.parse_args()
    for name in ("adult_dir", "mushroom_dir", "output_dir"):
        p = getattr(args, name)
        setattr(args, name, (p if p.is_absolute() else ROOT / p).resolve())
    if args.cv_jobs == 0:
        parser.error("--cv-jobs 不能为 0")
    return args


def make_classifiers():
    """与 Adult 现有模型完全一致，包含核函数、概率计算和随机种子。"""
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }


def metrics(y, pred, prob):
    if not (len(y) == len(pred) == len(prob)):
        raise ValueError("标签、预测、概率的长度不一致")
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError("概率不是有限的 [0, 1] 数值")
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, pos_label=1, zero_division=0),
        "Recall": recall_score(y, pred, pos_label=1, zero_division=0),
        "F1": f1_score(y, pred, pos_label=1, zero_division=0),
        "ROC-AUC": roc_auc_score(y, prob),
        "Kappa": cohen_kappa_score(y, pred),
    }


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def source_hashes(adult, mushroom):
    """记录原项目的所有文件，结束时确认没有改写原代码、数据或结果。"""
    records = {}
    for folder in (adult, mushroom):
        if not folder.is_dir():
            raise FileNotFoundError(f"找不到原项目：{folder}")
        for p in sorted(folder.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                records[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return records


def load_adult(args, classifiers):
    """复用固定方案二的正式成绩，不挑选每个模型的最高测试分数。"""
    prepared = args.adult_dir / "outputs/preprocessing/scheme2.joblib"
    trained = args.adult_dir / "outputs/models/scheme2.joblib"
    table_path = (
        args.adult_dir / "outputs/evaluation/adult_comparison"
        / "adult_all_schemes_results.csv"
    )
    for p in (prepared, trained, table_path):
        if not p.is_file():
            raise FileNotFoundError(f"缺少 Adult 正式结果：{p}")
    state, saved = joblib.load(prepared), joblib.load(trained)
    if not saved["y_test"].equals(state["y_test"]):
        raise ValueError("Adult 保存的模型与预处理结果测试标签不一致")
    original = pd.read_csv(table_path)
    result_rows = []
    if {r["name"] for r in saved["runs"]} != set(MODEL_NAMES):
        raise ValueError("Adult 方案二没有包含预期的四个模型")
    for run in saved["runs"]:
        name = run["name"]
        actual = run["pipe"].named_steps["model"].get_params(deep=False)
        expected = classifiers[name].get_params(deep=False)
        if actual != expected:
            raise ValueError(f"{name} 的保存参数与统一基线不一致：{actual} != {expected}")
        test = metrics(saved["y_test"], run["pred"], run["prob"])
        row = original[
            (original["Scheme"] == "Scheme 2") & (original["Model"] == name)
        ].iloc[0]
        if not all(np.isclose(row[k], v, rtol=1e-10, atol=1e-12)
                   for k, v in test.items()):
            raise ValueError(f"Adult {name} 的预测与原指标表不一致")
        cv_mean = float(run["cv_accuracy"])
        if not np.isclose(row["CV Accuracy (mean)"], cv_mean):
            raise ValueError(f"Adult {name} 的 CV 均值与指标表不一致")
        result_rows.append({
            "Dataset": "Adult", "Preprocessing": "Scheme 2",
            "Model": name, "CV folds": 10, "CV Accuracy (mean)": cv_mean,
            **test, "Train rows": len(state["y_train"]),
            "Test rows": len(saved["y_test"]),
            "Source": str(trained),
        })
    return result_rows


def run_mushroom(args, classifiers, out):
    """十折验证只使用训练集，填补器与编码器始终位于被验证的 Pipeline 内。"""
    module_path = args.mushroom_dir / "tree_models.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"缺少 Mushroom 数据读取模块：{module_path}")
    spec = importlib.util.spec_from_file_location("_original_mushroom_tree", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X, y = module.load_mushroom_data()
    if set(y.unique()) != {0, 1}:
        raise ValueError("Mushroom 标签必须为 edible=0、poisonous=1")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    if set(X_train.index) & set(X_test.index):
        raise ValueError("训练集与测试集的原始行号重叠")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # 导出每条记录的划分及验证折，便于复核。测试集不属于任何 CV 验证折。
    assignments = pd.DataFrame({
        "original_row": X.index, "y": y.to_numpy(),
        "split": "test", "cv_validation_fold": pd.NA,
    }).set_index("original_row")
    assignments.loc[X_train.index, "split"] = "train"
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        if set(fit_idx) & set(val_idx):
            raise ValueError(f"第 {fold} 折的训练与验证行重叠")
        assignments.loc[X_train.iloc[val_idx].index, "cv_validation_fold"] = fold
    assignments.to_csv(out / "split_and_cv_assignments.csv")

    prep = module.build_preprocessor(X_train)
    rows, folds, predictions, matrices, saved_runs = [], [], [], [], []
    print(
        f"Mushroom: train={len(y_train)}, test={len(y_test)}, "
        f"CV=10, seed=42, cv_jobs={args.cv_jobs}", flush=True
    )
    for name in MODEL_NAMES:
        pipe = Pipeline([
            ("preprocessing", clone(prep)),
            ("classifier", clone(classifiers[name])),
        ])
        print(f"开始 {name}：十折交叉验证", flush=True)
        started = perf_counter()
        cv_result = cross_validate(
            pipe, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy"}, n_jobs=args.cv_jobs,
            error_score="raise",
        )
        scores = cv_result["test_accuracy"]
        if len(scores) != 10 or not np.isfinite(scores).all():
            raise ValueError(f"{name} 未产生十个有效验证分数")
        for i, score in enumerate(scores):
            folds.append({
                "Dataset": "Mushroom", "Model": name, "Fold": i + 1,
                "Accuracy": float(score),
                "Fit seconds": float(cv_result["fit_time"][i]),
            })
        print(f"{name}：验证完成，拟合完整训练集并预测测试集", flush=True)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        classes = list(pipe.classes_)
        prob = pipe.predict_proba(X_test)[:, classes.index(1)]
        test = metrics(y_test, pred, prob)
        rows.append({
            "Dataset": "Mushroom", "Preprocessing": "Original missing category",
            "Model": name, "CV folds": 10,
            "CV Accuracy (mean)": float(scores.mean()), **test,
            "Train rows": len(y_train), "Test rows": len(y_test),
            "Source": str(args.mushroom_dir / "output/data/mushroom_modeling_base.csv"),
        })
        predictions.append(pd.DataFrame({
            "original_row": X_test.index, "Model": name,
            "y_true": y_test.to_numpy(), "y_pred": pred, "p_positive": prob,
        }))
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        matrices.append((name, cm))
        saved_runs.append({
            "name": name, "pipe": pipe, "cv_scores": scores,
            "pred": pred, "prob": prob,
        })
        print(
            f"{name}: CV={scores.mean():.6f}, test Accuracy={test['Accuracy']:.6f}, "
            f"F1={test['F1']:.6f}, AUC={test['ROC-AUC']:.6f}, "
            f"time={perf_counter()-started:.1f}s", flush=True
        )
    pd.DataFrame(folds).to_csv(out / "cv_scores_10fold.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out / "test_predictions.csv", index=False
    )
    pd.DataFrame(rows).to_csv(out / "metrics_10fold_holdout.csv", index=False)
    joblib.dump(
        {"y_test": y_test, "test_indices": X_test.index, "runs": saved_runs},
        out / "models.joblib", compress=3
    )
    matrix_rows = []
    for name, cm in matrices:
        for actual in (0, 1):
            for predicted in (0, 1):
                matrix_rows.append({
                    "Model": name, "Actual": actual, "Predicted": predicted,
                    "Count": int(cm[actual, predicted]),
                })
    pd.DataFrame(matrix_rows).to_csv(out / "confusion_matrices.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, cm) in zip(axes.flat, matrices):
        ax.imshow(cm, cmap="Blues")
        for r in (0, 1):
            for c in (0, 1):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual",
               title=name, xticklabels=["Edible", "Poisonous"],
               yticklabels=["Edible", "Poisonous"])
    fig.suptitle("Mushroom: unified baseline, holdout test")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=200)
    plt.close(fig)
    return rows, {
        "rows": len(X), "features": X.shape[1],
        "train_rows": len(X_train), "test_rows": len(X_test),
        "question_mark_count": int((X.astype("string") == "?").sum().sum()),
        "actual_nan_count": int(X.isna().sum().sum()),
        "positive_label": "poisonous=1",
        "preprocessing_note": (
            "沿用原读取方式。? 作为独立类别编码；SimpleImputer 不会填补普通字符串 ?。"
        ),
    }


def plot_comparison(frame, out):
    colors = {"Adult": "#264B70", "Mushroom": "#16857E"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric in zip(axes.flat, ("Accuracy", "F1", "Recall", "ROC-AUC")):
        x = np.arange(4)
        for j, dataset in enumerate(("Adult", "Mushroom")):
            data = frame[frame["Dataset"] == dataset].set_index("Model")
            bars = ax.bar(
                x + (j - 0.5)*0.36, data.loc[list(MODEL_NAMES), metric],
                width=0.36, label=dataset, color=colors[dataset],
            )
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set(xticks=x, xticklabels=["Tree", "Forest", "Logistic", "SVM (RBF)"],
               ylim=(0, 1.12), title=metric, ylabel="Score")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Adult Scheme 2 vs Mushroom: identical classifier parameters")
    fig.tight_layout()
    fig.savefig(out / "cross_dataset_metrics.png", dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(4)
    for j, dataset in enumerate(("Adult", "Mushroom")):
        data = frame[frame["Dataset"] == dataset].set_index("Model")
        bars = ax.bar(
            x+(j-0.5)*0.36, data.loc[list(MODEL_NAMES), "CV Accuracy (mean)"],
            width=0.36, label=dataset, color=colors[dataset],
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=["Tree", "Forest", "Logistic", "SVM (RBF)"],
           ylim=(0, 1.12), ylabel="Mean validation accuracy",
           title="Training-set stratified 10-fold CV")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "cross_dataset_cv_accuracy.png", dpi=200)
    plt.close(fig)


def main():
    args = options()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "mushroom"
    out.mkdir(parents=True, exist_ok=True)
    before = source_hashes(args.adult_dir, args.mushroom_dir)
    write_json(args.output_dir / "run_manifest.json", {"status": "running"})
    classifiers = make_classifiers()
    config = {
        "seed": 42, "cv_folds": 10, "cv_shuffle": True,
        "test_size": 0.2, "stratified_holdout": True,
        "cv_scoring": "accuracy", "adult_scheme": 2,
        "classifiers": {n: m.get_params(deep=False) for n, m in classifiers.items()},
        "versions": {
            "python": sys.version, "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
        "notes": [
            "Adult 复用正式结果，并逐一验证保存模型的全部分类器参数。",
            "Mushroom 新增固定参数十折验证及留出评估，原五折调参结果保留。",
            "两个数据集保留各自预处理，不把全部表现差异归因于数据集。",
            "Adult 原有重复记录风险仍需在报告中说明。",
        ],
    }
    write_json(args.output_dir / "experiment_config.json", config)
    adult_rows = load_adult(args, classifiers)
    pd.DataFrame(adult_rows).to_csv(
        args.output_dir / "adult_scheme2_reference.csv", index=False
    )
    mushroom_rows, data_info = run_mushroom(args, classifiers, out)
    combined = pd.DataFrame(adult_rows + mushroom_rows)
    if len(combined) != 8 or combined[["Dataset", "Model"]].duplicated().any():
        raise ValueError("汇总表没有形成两个数据集乘四种模型的八行结果")
    combined.to_csv(args.output_dir / "cross_dataset_metrics.csv", index=False)
    plot_comparison(combined, args.output_dir)
    after = source_hashes(args.adult_dir, args.mushroom_dir)
    if before != after:
        raise RuntimeError("运行期间原项目文件发生变化，请检查后再使用结果")
    write_json(args.output_dir / "source_preservation.json", {
        "original_projects_unchanged": True,
        "checked_files": len(before), "sha256": before,
    })
    write_json(args.output_dir / "run_manifest.json", {
        "status": "complete", "mushroom": data_info,
        "classifier_parameters_match_adult": True,
        "cv_folds": 10, "mushroom_cv_score_rows": 40,
        "comparison_rows": 8, "original_projects_unchanged": True,
        "adult_retrained": False,
    })
    print("\n统一基线实验完成：", args.output_dir, flush=True)
    print(combined[[
        "Dataset", "Model", "CV Accuracy (mean)", "Accuracy", "F1", "ROC-AUC"
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()


"""四个模型的交叉验证网格搜索与测试集评估。

训练/测试划分与 evaluate_models.py 完全一致（同种子、同分层），
因此调参前后的指标可以直接比较。网格搜索只在训练集上做交叉验证，
测试集只用于最终评估。
"""

from pathlib import Path
import time

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)

from linear_svm_models import get_linear_models, get_linear_param_grids
from tree_models import (
    RANDOM_SEED,
    get_tree_models,
    get_tree_param_grids,
    load_mushroom_data,
)

PROJECT_FOLDER = Path(__file__).resolve().parent
TABLES_FOLDER = PROJECT_FOLDER / "output" / "tables"

TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
REFIT = "accuracy"

SAFE_MODEL_NAMES = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
}


def split_train_test():
    """与 evaluate_models.prepare_data() 相同的划分。"""

    X, y = load_mushroom_data()
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )


def _format_params(params):
    """把参数字典整理成一行短文本，便于写入表格。"""

    short = {
        key.replace("classifier__estimator__", "svc__")
        .replace("classifier__", ""): value
        for key, value in params.items()
    }
    return ", ".join(f"{key}={value}" for key, value in sorted(short.items()))


def run_grid_search(model, param_grid, X_train, y_train):
    """对一个模型执行 GridSearchCV，返回 (search, 耗时秒)。"""

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=SCORING,
        refit=REFIT,
        cv=cv,
        n_jobs=-1,
    )

    start = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - start

    return search, elapsed


def _cv_metric(search, metric):
    values = search.cv_results_[f"mean_test_{metric}"]
    return float(values[search.best_index_])


def _cv_metric_std(search, metric):
    values = search.cv_results_[f"std_test_{metric}"]
    return float(values[search.best_index_])


def _save_cv_results(model_name, search):
    """逐组合的 CV 结果，便于复核调参过程。"""

    table = pd.DataFrame(search.cv_results_)
    keep = [
        column
        for column in table.columns
        if column.startswith("param_") or column.startswith("mean_test_")
        or column.startswith("std_test_") or column == "rank_test_accuracy"
    ]
    path = TABLES_FOLDER / f"tuning_cv_results_{SAFE_MODEL_NAMES[model_name]}.csv"
    table[keep].to_csv(path, index=False)
    return path


def run_tuning(X_train, y_train, X_test, y_test):
    """在给定划分上执行四个模型的网格搜索并写出结果表。"""

    from evaluate_models import evaluate_model  # 复用统一指标计算，避免第二套定义

    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)

    models = {
        **get_tree_models(X_train),
        **get_linear_models(X_train),
    }
    grids = {
        **get_tree_param_grids(),
        **get_linear_param_grids(),
    }

    print(f"交叉验证：{CV_FOLDS} 折，打分指标 {list(SCORING)}，refit={REFIT}")
    print(f"训练集：{X_train.shape}，测试集：{X_test.shape}")

    summary_rows = []
    baseline_rows = []
    cv_paths = []

    for model_name in SAFE_MODEL_NAMES:
        if model_name not in models:
            raise KeyError(f"未找到模型 {model_name!r}。")
        if model_name not in grids:
            raise KeyError(f"未找到 {model_name!r} 的调参网格。")

        n_candidates = len(list(ParameterGrid(grids[model_name])))
        print(
            f"\n[{model_name}] 候选组合 {n_candidates} 个，"
            f"共 {n_candidates * CV_FOLDS} 次拟合"
        )

        search, elapsed = run_grid_search(
            models[model_name], grids[model_name], X_train, y_train
        )

        # 基线：同一 Pipeline 的默认参数，用于对比调参收益
        baseline = clone(models[model_name]).fit(X_train, y_train)
        baseline_metrics = evaluate_model(model_name, baseline, X_test, y_test)

        best_metrics = evaluate_model(
            model_name, search.best_estimator_, X_test, y_test
        )

        summary_rows.append({
            "model": model_name,
            "candidates": n_candidates,
            "cv_folds": CV_FOLDS,
            "refit_metric": REFIT,
            "best_params": _format_params(search.best_params_),
            "cv_accuracy_mean": _cv_metric(search, "accuracy"),
            "cv_accuracy_std": _cv_metric_std(search, "accuracy"),
            "cv_f1_mean": _cv_metric(search, "f1"),
            "cv_f1_std": _cv_metric_std(search, "f1"),
            "cv_roc_auc_mean": _cv_metric(search, "roc_auc"),
            "cv_roc_auc_std": _cv_metric_std(search, "roc_auc"),
            "test_accuracy": best_metrics["accuracy"],
            "test_precision": best_metrics["precision"],
            "test_recall": best_metrics["recall"],
            "test_f1": best_metrics["f1"],
            "test_roc_auc": best_metrics["roc_auc"],
            "search_sec": round(elapsed, 2),
        })

        baseline_rows.append({
            "model": model_name,
            "baseline_test_accuracy": baseline_metrics["accuracy"],
            "tuned_test_accuracy": best_metrics["accuracy"],
            "delta_accuracy": best_metrics["accuracy"] - baseline_metrics["accuracy"],
            "baseline_test_f1": baseline_metrics["f1"],
            "tuned_test_f1": best_metrics["f1"],
            "delta_f1": best_metrics["f1"] - baseline_metrics["f1"],
            "baseline_test_roc_auc": baseline_metrics["roc_auc"],
            "tuned_test_roc_auc": best_metrics["roc_auc"],
            "delta_roc_auc": best_metrics["roc_auc"] - baseline_metrics["roc_auc"],
        })

        cv_paths.append(_save_cv_results(model_name, search))

        print(f"  最佳参数：{_format_params(search.best_params_)}")
        print(
            f"  CV  accuracy={_cv_metric(search, 'accuracy'):.6f}"
            f" f1={_cv_metric(search, 'f1'):.6f}"
            f" roc_auc={_cv_metric(search, 'roc_auc'):.6f}"
        )
        print(
            f"  测试集 accuracy={best_metrics['accuracy']:.6f}"
            f" f1={best_metrics['f1']:.6f}"
            f" roc_auc={best_metrics['roc_auc']:.6f}"
            f"（耗时 {elapsed:.1f}s）"
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = TABLES_FOLDER / "tuning_best_params.csv"
    summary.to_csv(summary_path, index=False)

    comparison = pd.DataFrame(baseline_rows)
    comparison_path = TABLES_FOLDER / "tuning_baseline_vs_tuned.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\n" + "=" * 70)
    print("网格搜索汇总（按 CV accuracy 排序）")
    print("=" * 70)
    print(
        summary.sort_values("cv_accuracy_mean", ascending=False)[
            ["model", "best_params", "cv_accuracy_mean", "test_accuracy", "test_f1"]
        ].to_string(index=False)
    )

    print(f"\n最佳参数与评估：{summary_path}")
    print(f"基线对比：{comparison_path}")
    for path in cv_paths:
        print(f"逐组合 CV 结果：{path}")

    return summary


def main():
    print("=" * 70)
    print("MUSHROOM GRID SEARCH")
    print("=" * 70)
    X_train, X_test, y_train, y_test = split_train_test()
    run_tuning(X_train, y_train, X_test, y_test)
    print("\n" + "=" * 70)
    print("网格搜索完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

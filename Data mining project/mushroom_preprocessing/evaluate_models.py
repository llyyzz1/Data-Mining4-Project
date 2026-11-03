"""四个模型的统一测试集评估与绘图脚本，并调用网格搜索与 Apriori。

预处理器只在训练集上拟合；测试集仅用于最终预测与指标计算，
不参与模型选择、阈值调整或超参数确定。
"""

from pathlib import Path

import matplotlib

# 无显示环境下（服务器/纯命令行）也能保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree


# 数据读取方式与预处理流程。
from linear_svm_models import get_linear_models
from tree_models import RANDOM_SEED, get_tree_models, load_mushroom_data

PROJECT_FOLDER = Path(__file__).resolve().parent
OUTPUT_FOLDER = PROJECT_FOLDER / "output"
FIGURES_FOLDER = OUTPUT_FOLDER / "figures"
TABLES_FOLDER = OUTPUT_FOLDER / "tables"

POSITIVE_LABEL = 1                    # poisonous = 1，作为正类
NEGATIVE_LABEL = 0                    # edible    = 0
CLASS_NAMES = ["edible", "poisonous"]  # 索引位置与标签 0 / 1 一一对应

TEST_SIZE = 0.2
RANDOM_STATE = RANDOM_SEED
FIGURE_DPI = 200

# 是否在基线评估之后执行交叉验证网格搜索（model_tuning.py）
RUN_TUNING = True
# 是否执行 Apriori 关联分析（association_rules.py）
RUN_APRIORI = True

# 模型英文名 -> 安全文件名（空格替换为下划线，避免路径问题）
SAFE_MODEL_NAMES = {
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "Linear SVM": "linear_svm",
}

# 报告文字用的模型短名（图表标题与图例使用）
SHORT_MODEL_NAMES = {
    "Decision Tree": "Decision Tree",
    "Random Forest": "Random Forest",
    "Logistic Regression": "Logistic Regression",
    "Linear SVM": "Linear SVM",
}

# 绘图配色（每个模型固定一种颜色，保证混淆矩阵与 ROC 图颜色一致）
MODEL_COLORS = {
    "Decision Tree": "#4C78A8",
    "Random Forest": "#287271",
    "Logistic Regression": "#C8553D",
    "Linear SVM": "#7B5EA7",
}


def ensure_output_folders() -> None:
    """创建 output/figures/ 与 output/tables/（已存在时不影响）。"""

    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
    TABLES_FOLDER.mkdir(parents=True, exist_ok=True)


def prepare_data():
    """读取数据并按 8:2 分层划分训练集与测试集。

    特征列与标签列全部由 tree_models.load_mushroom_data() 决定：
         标签列   : "class"（tree_models.TARGET_COLUMN）
         编码方式 : e -> 0（edible），p -> 1（poisonous，正类）
         特征     : 其余 21 个分类属性

    mushroom_modeling_base.csv 中 stalk-root 的缺失值写成 "?"，
    现有脚本用 pd.read_csv 默认参数读取，"?" 会被当作一个普通类别
    （而不是 NaN），再由 Pipeline 内的 OneHotEncoder 变成独立的一列。
    """

    X, y = load_mushroom_data()

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"load_mushroom_data() 返回的特征不是 DataFrame，"
            f"而是 {type(X)}；本脚本依赖列名，先确认现有脚本未被改动。"
        )

    # 检查标签编码：load_mushroom_data() 已把 e/p 映射为 0/1，
    # 这里只做“是否真的是二分类 0/1”的校验，不做静默修补。
    unique_labels = np.unique(np.asarray(y))
    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(
            "标签中出现了 0/1 之外的值，无法按 poisonous=1 作为正类处理。"
            f"实际取值为：{unique_labels.tolist()}；"
            f"请检查数据文件的 class 列（期望只有 e / p）。"
        )
    if not {0, 1}.issubset(set(unique_labels.tolist())):
        raise ValueError(
            "标签只有一个类别，无法进行二分类评估。"
            f"实际取值为：{unique_labels.tolist()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    if not {0, 1}.issubset(set(np.unique(np.asarray(y_train)).tolist())):
        raise ValueError(
            "训练集只包含一个类别，无法训练二分类模型。"
            f"训练集标签取值：{np.unique(np.asarray(y_train)).tolist()}"
        )

    return X_train, X_test, y_train, y_test


def build_all_models(X_train):
    """从现有脚本导入模型构造函数，返回四个 Pipeline。

    两个构造函数都需要传入 X 来构建预处理器（按 X 的列名建立
    ColumnTransformer），因此这里传入训练集特征，避免测试信息泄漏
    到预处理流程。
    """

    models = {}

    tree_models = get_tree_models(X_train)
    linear_models = get_linear_models(X_train)

    models.update(tree_models)
    models.update(linear_models)

    expected_names = list(SAFE_MODEL_NAMES.keys())
    missing = [name for name in expected_names if name not in models]
    if missing:
        raise KeyError(
            f"预期存在但未找到的模型：{missing}。"
            f"实际得到的模型：{list(models.keys())}；"
            "请确认 tree_models.py / linear_svm_models.py 未被改动。"
        )

    for name in expected_names:
        model = models[name]
        if not isinstance(model, Pipeline):
            raise TypeError(
                f"模型 {name!r} 不是 sklearn Pipeline，而是 {type(model)}；"
                "本脚本依赖 Pipeline 的 named_steps['preprocessing'] 与 "
                "named_steps['classifier']。"
            )
        steps = list(model.named_steps.keys())
        if steps != ["preprocessing", "classifier"]:
            raise ValueError(
                f"模型 {name!r} 的 Pipeline 步骤是 {steps}，"
                "但本脚本预期为 ['preprocessing', 'classifier']。"
            )

    return models


def get_feature_names(pipeline, model_name):
    """从已拟合的 Pipeline 的 preprocessing 步骤取 one-hot 后的特征名。"""

    preprocessor = pipeline.named_steps["preprocessing"]
    if not hasattr(preprocessor, "get_feature_names_out"):
        raise AttributeError(
            f"模型 {model_name!r} 的 preprocessing 步骤"
            f"（{type(preprocessor)}）没有 get_feature_names_out()，"
            "无法取得特征名。"
        )
    return np.asarray(preprocessor.get_feature_names_out(), dtype=object)


def get_positive_class_scores(pipeline, X, model_name):
    """取得正类（poisonous = 1）的概率分数。

    通过 classifier.classes_ 查找正类所在的概率列，不假定
    “第 2 列就是正类”；概率无效时直接报错，不伪造分数。
    """

    classifier = pipeline.named_steps["classifier"]

    if not hasattr(classifier, "predict_proba"):
        raise AttributeError(
            f"模型 {model_name!r} 的 classifier（{type(classifier).__name__}）"
            "没有 predict_proba()，本脚本不会用其他分数冒充概率。"
        )

    if not hasattr(classifier, "classes_"):
        raise AttributeError(
            f"模型 {model_name!r} 的 classifier 没有 classes_ 属性，"
            "无法确认正类对应的概率列。"
        )

    classes = list(np.asarray(classifier.classes_).tolist())
    if POSITIVE_LABEL not in classes:
        raise ValueError(
            f"模型 {model_name!r} 的 classes_ = {classes}，"
            f"其中找不到正类标签 {POSITIVE_LABEL}，无法定位正类概率列。"
        )

    column_index = classes.index(POSITIVE_LABEL)
    proba = np.asarray(pipeline.predict_proba(X))

    if proba.ndim != 2 or proba.shape[1] != len(classes):
        raise ValueError(
            f"模型 {model_name!r} 的 predict_proba 输出形状为 {proba.shape}，"
            f"与 classes_（{classes}）不匹配。"
        )

    scores = proba[:, column_index].astype(float)

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            f"模型 {model_name!r} 的正类概率中含有非有限值（NaN / inf），"
            "该模型无法给出可靠的概率分数。"
        )
    if scores.min() < -1e-12 or scores.max() > 1 + 1e-12:
        raise ValueError(
            f"模型 {model_name!r} 的正类概率超出 [0, 1] 区间"
            f"（min={scores.min()}, max={scores.max()}），"
            "不是有效的概率，停止该模型的 AUC 计算。"
        )

    # 概率只有一个唯一值时，ROC 曲线没有意义（AUC 无定义）
    if np.unique(scores).size < 2:
        return scores, (
            f"模型只输出单一概率值 {scores[0]!r}，正类分数没有区分度，"
            "ROC-AUC 无定义，已跳过该模型的 AUC 计算。"
        )

    return scores, None


def evaluate_model(name, pipeline, X_test, y_test):
    """计算单个模型在测试集上的指标。

    正类固定为 poisonous = 1，所有二分类指标显式传入
    pos_label=POSITIVE_LABEL 与 zero_division=0。
    """

    pred = np.asarray(pipeline.predict(X_test))
    scores, skip_reason = get_positive_class_scores(pipeline, X_test, name)

    roc_auc = np.nan
    auc_note = ""
    if skip_reason is None:
        roc_auc = float(roc_auc_score(y_test, scores, labels=[0, 1]))
        auc_note = "ROC-AUC 使用 predict_proba 的正类概率。"
    else:
        auc_note = skip_reason

    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(
            precision_score(
                y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0
            )
        ),
        "f1": float(
            f1_score(y_test, pred, pos_label=POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": roc_auc,
        "predictions": pred,
        "scores": scores,
        "auc_note": auc_note,
        "confusion_matrix": confusion_matrix(
            y_test, pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
        ),
    }


def draw_confusion_matrix_figure(
    matrix,
    title,
    colorbar_label,
    value_formatter,
    highlight_threshold,
):
    """绘制一张混淆矩阵图并返回 Figure。

    布局说明：色条与标题位置是手工指定的（add_axes + suptitle），
    因为 fig.colorbar(ax=...) + tight_layout 会把标题挤出色条以外的区域，
    导致标题被裁掉。这里用明确的位置保证标题完整可见。
    """

    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_axes([0.14, 0.13, 0.60, 0.76])
    cax = fig.add_axes([0.80, 0.13, 0.04, 0.76])

    image = ax.imshow(matrix, cmap="Blues")
    # imshow 默认按等比例显示，会与上面的坐标框互相冲突，这里按坐标框拉伸
    ax.set_aspect("auto")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1], labels=CLASS_NAMES)
    ax.set_yticks([0, 1], labels=CLASS_NAMES)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                value_formatter(value),
                ha="center",
                va="center",
                color="white" if value > highlight_threshold else "black",
                fontsize=12,
            )

    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_label(colorbar_label)

    # 标题放在整张图的顶部居中，保证不会被色条裁掉
    fig.suptitle(title, fontsize=13, y=0.965)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    return fig


def plot_confusion_matrices(name, matrix):
    """为单个模型绘制原始数量版与归一化版混淆矩阵。

    类别顺序固定为 edible、poisonous；标题与坐标轴均为英文。
    """

    safe_name = SAFE_MODEL_NAMES[name]
    short_name = SHORT_MODEL_NAMES[name]

    count_threshold = matrix.max() / 2 if matrix.max() > 0 else 0
    fig = draw_confusion_matrix_figure(
        matrix=matrix,
        title=f"Confusion Matrix - {short_name}",
        colorbar_label="Count",
        value_formatter=lambda value: str(int(value)),
        highlight_threshold=count_threshold,
    )
    raw_path = FIGURES_FOLDER / f"confusion_matrix_{safe_name}.png"
    fig.savefig(raw_path, dpi=FIGURE_DPI)
    plt.close(fig)

    # 按真实类别行归一化，即每一类的召回率分布
    row_totals = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(row_totals > 0, matrix / row_totals, 0.0)

    fig = draw_confusion_matrix_figure(
        matrix=normalized,
        title=f"Normalized Confusion Matrix - {short_name}",
        colorbar_label="Proportion of True Class",
        value_formatter=lambda value: f"{value:.3f}",
        highlight_threshold=0.5,
    )
    normalized_path = (
        FIGURES_FOLDER / f"confusion_matrix_{safe_name}_normalized.png"
    )
    fig.savefig(normalized_path, dpi=FIGURE_DPI)
    plt.close(fig)

    raw_table = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in CLASS_NAMES],
        columns=[f"predicted_{label}" for label in CLASS_NAMES],
    )
    raw_table.index.name = "actual_label"
    raw_csv = TABLES_FOLDER / f"confusion_matrix_{safe_name}.csv"
    raw_table.to_csv(raw_csv)

    normalized_table = pd.DataFrame(
        normalized,
        index=[f"actual_{label}" for label in CLASS_NAMES],
        columns=[f"predicted_{label}" for label in CLASS_NAMES],
    )
    normalized_table.index.name = "actual_label"
    normalized_csv = (
        TABLES_FOLDER / f"confusion_matrix_{safe_name}_normalized.csv"
    )
    normalized_table.round(6).to_csv(normalized_csv)

    return [raw_path, normalized_path, raw_csv, normalized_csv]


def plot_roc_comparison(results):
    """绘制四个模型的 ROC 对照图，图例标注各自 AUC。"""

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    plotted_any = False
    for name, result in results.items():
        if not np.isfinite(result["roc_auc"]):
            print(
                f"  [跳过 ROC] {name}：{result['auc_note']}"
            )
            continue

        # 所有模型都用 predict_proba 的正类概率，方向一致
        # （分数越高越可能是 poisonous）。
        false_positive_rate, true_positive_rate, _ = roc_curve(
            result["y_test"],
            result["scores"],
            pos_label=POSITIVE_LABEL,
        )
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=MODEL_COLORS[name],
            linewidth=2,
            label=f"{SHORT_MODEL_NAMES[name]} (AUC = {result['roc_auc']:.4f})",
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print("  [跳过 ROC] 没有任何模型能提供可用的正类概率，未生成 ROC 对照图。")
        return None

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Chance Level",
    )
    ax.set_title("ROC Curve Comparison (Test Set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path = FIGURES_FOLDER / "roc_curve_comparison.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)

    return output_path


def plot_model_importance(name, pipeline):
    """用模型自带的重要度/系数核对特征名与权重长度是否一致。

    返回值只用于控制台核对与报告，不额外画图。
    """

    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(pipeline, name)

    if hasattr(classifier, "feature_importances_"):
        weights = np.asarray(classifier.feature_importances_)
        kind = "feature_importances_"
    elif hasattr(classifier, "coef_"):
        weights = np.ravel(np.asarray(classifier.coef_))
        kind = "coef_"
    else:
        return None

    if len(feature_names) != weights.shape[0]:
        raise ValueError(
            f"模型 {name!r} 的特征名数量（{len(feature_names)}）"
            f"与 {kind} 长度（{weights.shape[0]}）不一致，"
            "说明特征名与模型权重对应不上。"
        )

    order = np.argsort(weights)[::-1][:5]
    top_features = [
        (str(feature_names[i]), float(weights[i])) for i in order
    ]
    return kind, len(feature_names), top_features


def plot_decision_tree_structure(name, pipeline):
    """绘制决策树结构图。

    直接绘制用于评估的已训练 Pipeline 中的决策树，保证图与原模型
    完全对应，不另建浅层替代模型。若树过大（叶子数超过显示上限），
    打印明确警告并跳过绘制。
    """

    classifier = pipeline.named_steps["classifier"]
    if not isinstance(classifier, DecisionTreeClassifier):
        print(f"  [跳过树结构图] {name!r} 的 classifier 不是 DecisionTreeClassifier。")
        return None

    feature_names = get_feature_names(pipeline, name)
    if len(feature_names) != classifier.n_features_in_:
        raise ValueError(
            f"决策树的特征名数量（{len(feature_names)}）"
            f"与 n_features_in_（{classifier.n_features_in_}）不一致，"
            "无法把特征名正确地标到树上。"
        )

    # 已拟合模型里的类别顺序必须与 edible / poisonous 的展示顺序一致
    classes = list(np.asarray(classifier.classes_).tolist())
    if classes != [NEGATIVE_LABEL, POSITIVE_LABEL]:
        raise ValueError(
            f"决策树的 classes_ = {classes}，"
            f"与预期的 [{NEGATIVE_LABEL}, {POSITIVE_LABEL}] 不一致；"
            "此时 class_names 的显示顺序会错位，因此停止绘制。"
        )

    depth = classifier.get_depth()
    leaves = classifier.get_n_leaves()
    print(
        f"  决策树实际规模：深度 {depth}，叶子 {leaves} 个"
        "（绘制的是用于评估的同一个已训练模型）。"
    )

    max_display_nodes = 200
    if leaves > max_display_nodes:
        print(
            f"  [跳过树结构图] 该树叶子数 {leaves} 超过 {max_display_nodes}，"
            "直接绘制会无法辨认；为避免把图误称为原模型结构，"
            "本脚本不自动另建浅层树。若需要展示用浅层树，请先确认。"
        )
        return None

    fig, ax = plt.subplots(figsize=(26, 13))
    plot_tree(
        classifier,
        feature_names=[str(f) for f in feature_names],
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(
        "Decision Tree Structure "
        f"(depth = {depth}, leaves = {leaves})"
    )
    fig.tight_layout()

    output_path = FIGURES_FOLDER / "decision_tree_structure.png"
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)

    return output_path


def main():
    ensure_output_folders()

    print("=" * 70)
    print("MUSHROOM MODEL EVALUATION (TEST SET)")
    print("=" * 70)
    print(f"random_state = {RANDOM_STATE}, test_size = {TEST_SIZE}")
    print(f"positive class = poisonous = {POSITIVE_LABEL}")
    print(f"figures -> {FIGURES_FOLDER}")
    print(f"tables  -> {TABLES_FOLDER}")

    X_train, X_test, y_train, y_test = prepare_data()

    print(f"\n训练集：{X_train.shape}，测试集：{X_test.shape}")
    print("训练集标签分布：", dict(pd.Series(y_train).value_counts().sort_index()))
    print("测试集标签分布：", dict(pd.Series(y_test).value_counts().sort_index()))

    models = build_all_models(X_train)

    print("\n" + "-" * 70)
    print("在训练集上拟合 Pipeline（测试集不参与拟合）")
    print("-" * 70)

    results = {}
    feature_checks = {}

    for name in SAFE_MODEL_NAMES:
        pipeline = clone(models[name])
        pipeline.fit(X_train, y_train)

        check = plot_model_importance(name, pipeline)
        if check is not None:
            kind, count, top_features = check
            feature_checks[name] = check
            print(
                f"\n[{name}] 已拟合；{kind} 长度 = {count}，"
                "与 one-hot 特征名数量一致。"
            )
            for feature, weight in top_features[:3]:
                print(f"    top: {feature} -> {weight:.4f}")
        else:
            print(f"\n[{name}] 已拟合；该模型没有 coef_ / feature_importances_，跳过核对。")

        result = evaluate_model(name, pipeline, X_test, y_test)
        result["y_test"] = np.asarray(y_test)
        result["pipeline"] = pipeline
        results[name] = result

    metrics_rows = []
    for name, result in results.items():
        metrics_rows.append({
            "model": name,
            "accuracy": round(result["accuracy"], 6),
            "precision": round(result["precision"], 6),
            "recall": round(result["recall"], 6),
            "f1": round(result["f1"], 6),
            "roc_auc": (
                round(result["roc_auc"], 6)
                if np.isfinite(result["roc_auc"])
                else np.nan
            ),
        })

    metrics_table = pd.DataFrame(metrics_rows)
    metrics_path = TABLES_FOLDER / "model_metrics.csv"
    metrics_table.to_csv(metrics_path, index=False)

    print("\n" + "=" * 70)
    print("测试集指标（正类：poisonous = 1）")
    print("=" * 70)
    print(metrics_table.to_string(index=False))
    print(f"\n指标表已保存：{metrics_path}")

    for name, result in results.items():
        print(f"\n[{name}] {result['auc_note']}")

    print("\n" + "-" * 70)
    print("混淆矩阵（类别顺序：edible, poisonous）")
    print("-" * 70)

    for name, result in results.items():
        print(
            f"\n[{name}] 原始矩阵 (行=真实, 列=预测)：\n"
            f"{result['confusion_matrix']}"
        )
        for path in plot_confusion_matrices(name, result["confusion_matrix"]):
            print(f"  已保存：{path}")

    print("\n" + "-" * 70)
    print("ROC 曲线对照")
    print("-" * 70)
    roc_path = plot_roc_comparison(results)
    if roc_path is not None:
        print(f"ROC 对照图已保存：{roc_path}")

    print("\n" + "-" * 70)
    print("决策树结构图")
    print("-" * 70)
    tree_path = plot_decision_tree_structure(
        "Decision Tree",
        results["Decision Tree"]["pipeline"],
    )
    if tree_path is not None:
        print(f"决策树结构图已保存：{tree_path}")

    if RUN_TUNING:
        print("\n" + "-" * 70)
        print("交叉验证网格搜索（model_tuning.py）")
        print("-" * 70)
        from model_tuning import run_tuning

        run_tuning(X_train, y_train, X_test, y_test)

    if RUN_APRIORI:
        print("\n" + "-" * 70)
        print("Apriori 关联分析（association_rules.py）")
        print("-" * 70)
        from association_rules import run_apriori

        run_apriori()

    print("\n" + "=" * 70)
    print("全部评估完成。")
    print("=" * 70)


if __name__ == "__main__":
    main()
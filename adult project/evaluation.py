"""模型评估和可视化：读取保存的预测；--all 按顺序运行三个模块。"""

import subprocess
import sys

sys.dont_write_bytecode = True
from preprocessing import PROJECT_DIR, new_output, parse_options

# 演示阅读顺序：文件底部入口 -> run_evaluation -> evaluate_scheme1 -> compare_schemes。
# 三个方案的评估步骤相同，预处理和训练的差异已经由前两个模块完成。
# 本文件读取保存的预测，不调用 fit；十折交叉验证在 models.py 中执行。
# 指标统一以高收入为正类：1 表示 >50K，0 表示 <=50K。


def evaluate_scheme1(saved, OUT):
    """原方案一的指标和图表；OUT 是新增入口传入的输出目录。"""
    from pathlib import Path
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, cohen_kappa_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score, roc_curve,
    )
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    # 【演示重点 1：评估输入】saved 是模型阶段保存的字典，y_test 是真实测试标签。
    # runs 包含四个模型的结果；这里遍历已有结果，不重新训练或重新预测。
    y_test = saved["y_test"]
    rows, roc_data, matrices = [], {}, {}
    for result in saved["runs"]:
        # name：模型名；pipe：已训练的 Pipeline；pred：预测类别；prob：高收入概率。
        name, pipe, pred, prob = result["name"], result["pipe"], result["pred"], result["prob"]
        # cv_scores 是训练集十折验证得到的十个准确率，mean() 只负责汇总。
        # CV Accuracy 与下面的测试集 Accuracy 来自不同的评估过程。
        cv_scores = result["cv_scores"]
        # 【演示重点 2：指标】类别不平衡时，Accuracy 需要结合 Recall、F1、AUC 解读。
        # Precision = TP/(TP+FP)：预测为高收入的人中，实际高收入的比例。
        # Recall = TP/(TP+FN)：实际高收入的人中，成功识别的比例。
        # F1 是 Precision 与 Recall 的调和平均；默认针对正类 1 计算。
        # zero_division=0：分母为零、指标无定义时返回 0，不改变预测结果。
        rows.append({
            "Model": name, "CV Accuracy (mean)": cv_scores.mean(),
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            # AUC 使用连续概率 prob，而不是 0/1 类别 pred，衡量模型的排序能力。
            "ROC-AUC": roc_auc_score(y_test, prob),
            # Kappa 扣除随机一致的影响，是准确率之外的补充指标。
            "Kappa": cohen_kappa_score(y_test, pred),
        })
        # 【演示重点 3：ROC 与混淆矩阵】ROC 会改变概率阈值来计算 FPR 和 TPR。
        # 混淆矩阵行是实际类别、列是预测类别，按 0、1 排列：[[TN, FP], [FN, TP]]。
        roc_data[name] = roc_curve(y_test, prob)
        matrices[name] = confusion_matrix(y_test, pred)
        if name == "Decision Tree":
            # 从已有 Pipeline 取出树，以及编码后的特征名，保证图中列名与模型一致。
            tree_model = pipe.named_steps["model"]
            feature_names = pipe.named_steps["preprocess"].get_feature_names_out()

    # 整理四个模型的指标，并保存 CSV/Excel；按 F1 排序只用于展示测试结果。
    results = pd.DataFrame(rows).sort_values("F1", ascending=False)
    results.to_csv(OUT / "metrics.csv", index=False)
    results.to_excel(OUT / "metrics.xlsx", index=False)
    print(results.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    # ROC 横轴 FPR 是低收入误判为高收入的比例，纵轴 TPR 就是 Recall。
    # 第三个返回值是阈值数组，本图不直接显示，所以用 _ 接收。
    for name, (fpr, tpr, _) in roc_data.items():
        auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    # 对角虚线表示随机区分的参照，不代表模型实际的预测曲线。
    ax.plot([0, 1], [0, 1], "k--", alpha=.6)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Adult — Scheme 1 ROC Curves")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "roc_curves.png", dpi=300); plt.close(fig)

    # 2×2 子图分别展示四种模型；左下角 FN 是演示时要说明的高收入漏判人数。
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, cm) in zip(axes.flat, matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=300); plt.close(fig)

    # Tree diagram (top levels for readability).
    # max_depth=3 仅限制图中显示的层数，根节点深度为 0；不会更改训练好的树。
    fig = plt.figure(figsize=(24, 12))
    plot_tree(tree_model, feature_names=feature_names, class_names=["<=50K", ">50K"],
              filled=True, rounded=True, max_depth=3, fontsize=7)
    fig.suptitle("Adult — Decision Tree (Scheme 1, top 4 levels)")
    fig.savefig(OUT / "decision_tree.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"\nResults saved to: {OUT}")


def evaluate_scheme2(saved, OUT):
    """原方案二的评估语句，不重新拟合处理器或模型。"""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve)
    # 方案二使用自己的测试标签和四组预测；指标与图表含义同方案一。
    y_test = saved["y_test"]
    rows, curves, matrices = [], {}, {}
    for result in saved["runs"]:
        name, pipe, pred, prob = result["name"], result["pipe"], result["pred"], result["prob"]
        # 原方案二只保存十折准确率的均值，因此这里直接读取，不再次求均值或做 CV。
        cv_accuracy = result["cv_accuracy"]
        rows.append({"Model": name, "CV Accuracy (mean)": cv_accuracy,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred, zero_division=0),
                     "F1": f1_score(y_test, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_test, prob),
                     "Kappa": cohen_kappa_score(y_test, pred)})
        curves[name] = roc_curve(y_test, prob)
        matrices[name] = confusion_matrix(y_test, pred)
        if name == "Decision Tree":
            tree_model = pipe.named_steps["model"]
            tree_features = pipe.named_steps["preprocess"].get_feature_names_out()

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "scheme2_results.csv", index=False)
    try:
        results.to_excel(OUT / "scheme2_results.xlsx", index=False)
    except ImportError:
        pass
    print(results.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, _) in curves.items():
        auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Adult - Scheme 2 ROC Curves")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "scheme2_roc.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUT / "scheme2_confusion_matrices.png", dpi=300); plt.close(fig)

    plt.figure(figsize=(24, 12))
    plot_tree(tree_model, feature_names=tree_features, class_names=["<=50K", ">50K"],
              filled=True, rounded=True, max_depth=3, fontsize=7)
    plt.title("Adult - Scheme 2 Decision Tree (top levels)")
    plt.savefig(OUT / "scheme2_tree.png", dpi=300, bbox_inches="tight"); plt.close()
    print("Results saved to:", OUT)


def evaluate_scheme3(saved, OUT):
    """原方案三的评估和图表，原有参数及文件名保持不变。"""
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix, roc_curve)
    # 方案三沿用相同评估口径；保留缺失信息的处理已经在训练 Pipeline 中完成。
    y_test = saved["y_test"]
    rows, curves, matrices = [], {}, {}
    for result in saved["runs"]:
        name, pipe, pred, prob = result["name"], result["pipe"], result["pred"], result["prob"]
        # 与方案二一样，读取模型阶段保存的十折均值；当前函数不重新拟合任何对象。
        cv_accuracy = result["cv_accuracy"]
        rows.append({"Model": name, "CV Accuracy (mean)": cv_accuracy,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred, zero_division=0),
                     "F1": f1_score(y_test, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_test, prob),
                     "Kappa": cohen_kappa_score(y_test, pred)})
        curves[name] = roc_curve(y_test, prob)
        matrices[name] = confusion_matrix(y_test, pred)
        if name == "Decision Tree":
            tree_model = pipe.named_steps["model"]
            tree_features = pipe.named_steps["preprocess"].get_feature_names_out()

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "scheme3_results.csv", index=False)
    try:
        results.to_excel(OUT / "scheme3_results.xlsx", index=False)
    except ImportError:
        pass
    print(results.to_string(index=False))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (fpr, tpr, _) in curves.items():
        auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="Adult - Scheme 3 ROC Curves")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "scheme3_roc.png", dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(OUT / "scheme3_confusion_matrices.png", dpi=300); plt.close(fig)

    plt.figure(figsize=(24, 12))
    plot_tree(tree_model, feature_names=tree_features, class_names=["<=50K", ">50K"],
              filled=True, rounded=True, max_depth=3, fontsize=7)
    plt.title("Adult - Scheme 3 Decision Tree (top levels)")
    plt.savefig(OUT / "scheme3_tree.png", dpi=300, bbox_inches="tight"); plt.close()
    print("Results saved to:", OUT)


def compare_schemes(results_dir):
    """原 adult_compare.py，新增代码只绑定结果目录，保留其测试 F1 排名。"""
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    ROOT = Path(__file__).resolve().parents[1]
    RESULTS = ROOT / "modeling" / "results"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    RESULTS = results_dir
    OUT = RESULTS / "adult_comparison"
    OUT.mkdir(parents=True, exist_ok=True)

    # 【演示重点 4：方案比较】读取三个方案的指标表，加上 Scheme 列后纵向合并。
    # 每个方案四行，汇总共十二行；这里只比较 Adult，不包含 Mushroom 的成绩。
    files = {
        "Scheme 1": RESULTS / "adult_scheme1" / "metrics.csv",
        "Scheme 2": RESULTS / "scheme2" / "scheme2_results.csv",
        "Scheme 3": RESULTS / "scheme3" / "scheme3_results.csv",
    }
    frames = []
    for scheme, path in files.items():
        data = pd.read_csv(path)
        data.insert(0, "Scheme", scheme)
        frames.append(data)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(OUT / "adult_all_schemes_results.csv", index=False)
    combined.to_excel(OUT / "adult_all_schemes_results.xlsx", index=False)
    print(combined.to_string(index=False))

    sns.set_theme(style="whitegrid")
    # x 是模型，y 是所选指标，hue 用不同颜色区分方案，纵轴统一为 0 到 1。
    # 注意：方案一删除缺失记录后，测试人群与方案二、三不同，不能作严格控制比较。
    for metric, filename in [("Accuracy", "accuracy_comparison.png"),
                             ("F1", "f1_comparison.png"),
                             ("ROC-AUC", "roc_auc_comparison.png")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=combined, x="Model", y=metric, hue="Scheme", ax=ax)
        ax.set_title(f"Adult - {metric} Comparison")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUT / filename, dpi=300)
        plt.close(fig)

    # 先按测试 F1 排序，相同时再按测试 AUC 排序；iloc[0] 取当前观察排名第一行。
    # 演示时不要把它解释为已证明最优：模型选择应依据训练集 CV，测试集用于最终评估。
    best = combined.sort_values(["F1", "ROC-AUC"], ascending=False).iloc[0]
    print(f"\nBest overall row by F1 then ROC-AUC: {best['Scheme']} / {best['Model']}")
    print("Comparison files saved to:", OUT)


def run_evaluation(args):
    """评估只需模型阶段文件，不需要重新读取原始数据或重新运行预处理。"""
    import joblib
    import openpyxl  # 明确检查原有 Excel 导出依赖，不隐藏缺包错误。
    # 【演示重点 5：模块衔接】先确认三个模型结果存在，再逐个读取 joblib 字典。
    # args.work_dir 默认指向项目内 outputs，不依赖启动终端所在的目录。
    trained = args.work_dir / "models"
    for number in (1, 2, 3):
        if not (trained / f"scheme{number}.joblib").is_file():
            raise FileNotFoundError(f"缺少模型结果：{trained / f'scheme{number}.joblib'}；请先运行 models.py")
    output = new_output(args.work_dir / "evaluation")
    # enumerate 从 1 编号，把 scheme1/2/3 文件与各自的评估函数对应起来。
    for number, evaluate in enumerate((evaluate_scheme1, evaluate_scheme2, evaluate_scheme3), 1):
        name = "adult_scheme1" if number == 1 else f"scheme{number}"
        evaluate(joblib.load(trained / f"scheme{number}.joblib"), new_output(output / name))
    # 三个单独评估完成后，再生成 Adult 的汇总表和三个比较图。
    # Apriori 由 models.py 保存结果，当前评估入口不会读取或重新生成关联规则。
    compare_schemes(output)
    print(f"评估与图表已保存：{output}")


if __name__ == "__main__":
    # 【演示重点 6：运行入口】直接运行本文件时执行；其他文件 import 时不执行。
    # python evaluation.py：只评估。python evaluation.py --all：依次执行完整流程。
    args = parse_options("Adult 模型评估与可视化", allow_all=True)
    if args.all:
        # sys.executable 使用当前 Python 环境；两次子进程运行完成后，才进入下面的评估。
        # check=True 表示前一步失败就抛错停止，避免继续使用不完整的结果。
        # 使用同一个解释器和绝对路径；前一步失败会直接停止，不隐藏异常。
        for filename in ("preprocessing.py", "models.py"):
            subprocess.run([sys.executable, str(PROJECT_DIR / filename), "--data-dir", str(args.data_dir),
                            "--work-dir", str(args.work_dir)], check=True)
    run_evaluation(args)

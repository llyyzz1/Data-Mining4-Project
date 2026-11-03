"""数据预处理：原有处理方法放入普通函数，新增部分只负责路径和文件传递。"""

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
PROJECT_DIR = Path(__file__).resolve().parent


def parse_options(description, allow_all=False):
    """三个模块共用路径参数；相对路径以本文件所在目录为准。"""
    os.environ.setdefault("MPLBACKEND", "Agg")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "adult")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_DIR / "outputs")
    if allow_all:
        parser.add_argument("--all", action="store_true", help="依次运行预处理、模型和评估")
    args = parser.parse_args()
    for name in ("data_dir", "work_dir"):
        value = getattr(args, name)
        setattr(args, name, (value if value.is_absolute() else PROJECT_DIR / value).resolve())
    return args


def new_output(path):
    """创建或复用输出目录；重复运行时，同名结果文件会被本次结果覆盖。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_scheme1(input_dir, output_dir):
    """保留方案一自己的清洗和处理器，供模型在交叉验证中拟合。"""
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


    ROOT = Path(__file__).resolve().parents[1]
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    ROOT = PROJECT_DIR
    OUT = ROOT / "modeling" / "results" / "adult_scheme1"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    OUT = output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    # Load the same local Adult data used by the other schemes (adult.data + adult.test).
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    data_dir = ROOT / "adult"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    data_dir = input_dir
    train_raw = pd.read_csv(data_dir / "adult.data", names=columns, skipinitialspace=True, na_values="?")
    test_raw = pd.read_csv(data_dir / "adult.test", names=columns, skiprows=1, skipinitialspace=True, na_values="?")
    adult_df = pd.concat([train_raw, test_raw], ignore_index=True)
    X = adult_df.drop(columns="income")
    y = adult_df["income"]

    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    y = y.astype("string").str.strip().str.rstrip(".").replace({"?": pd.NA, "": pd.NA})
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].map({"<=50K": 0, ">50K": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scheme 1: remove rows with any missing feature, then encode/scale.
    train_ok, test_ok = X_train.notna().all(axis=1), X_test.notna().all(axis=1)
    X_train, y_train = X_train.loc[train_ok], y_train.loc[train_ok]
    X_test, y_test = X_test.loc[test_ok], y_test.loc[test_ok]
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                preprocess=preprocess, raw=adult_df)


def prepare_scheme2(input_dir, output_dir):
    """原方案二：保留其 object 转换，数值列为空的问题没有在此修复。"""
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

    ROOT = Path(__file__).resolve().parents[1]
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    ROOT = PROJECT_DIR
    DATA = ROOT / "adult"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    DATA = input_dir
    OUT = ROOT / "modeling" / "results" / "scheme2"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    OUT = output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "income"]

    # Use the complete local Adult data: adult.data + adult.test.
    train_raw = pd.read_csv(DATA / "adult.data", names=columns,
                            skipinitialspace=True, na_values="?")
    test_raw = pd.read_csv(DATA / "adult.test", names=columns, skiprows=1,
                           skipinitialspace=True, na_values="?")
    adult = pd.concat([train_raw, test_raw], ignore_index=True)
    X = adult.drop(columns="income")
    y = adult["income"]

    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = X[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    X = X.astype(object).where(X.notna(), np.nan)
    y = y.astype("string").str.strip().str.rstrip(".").replace({"?": pd.NA, "": pd.NA})
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].map({"<=50K": 0, ">50K": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()

    # Scheme 2: numeric median and categorical most-frequent imputation.
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    # Same four models and parameters as Scheme 1.
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                preprocess=preprocess)


def prepare_scheme3(input_dir, output_dir):
    """原方案三：保留原有填补和编码方法，包括原来的类型处理问题。"""
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

    ROOT = Path(__file__).resolve().parents[1]
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    ROOT = PROJECT_DIR
    DATA = ROOT / "adult"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    DATA = input_dir
    OUT = ROOT / "modeling" / "results" / "scheme3"
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    OUT = output_dir
    OUT.mkdir(parents=True, exist_ok=True)

    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "income"]

    # Use the complete local Adult data: adult.data + adult.test.
    train_raw = pd.read_csv(DATA / "adult.data", names=columns,
                            skipinitialspace=True, na_values="?")
    test_raw = pd.read_csv(DATA / "adult.test", names=columns, skiprows=1,
                           skipinitialspace=True, na_values="?")
    adult = pd.concat([train_raw, test_raw], ignore_index=True)
    X = adult.drop(columns="income")
    y = adult["income"]

    for col in X.select_dtypes(include=["object", "string", "category"]).columns:
        X[col] = X[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
    X = X.astype(object).where(X.notna(), np.nan)
    y = y.astype("string").str.strip().str.rstrip(".").replace({"?": pd.NA, "": pd.NA})
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].map({"<=50K": 0, ">50K": 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()

    # Scheme 3: numeric median plus missing indicators; categorical Missing category.
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])

    # Same four models and parameters as Scheme 1.
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                preprocess=preprocess)


def original_preprocessing(dataset, picture_dir):
    """原 pre_clean.py 的处理和绘图语句，仅包进函数并适配输入、图片路径。"""
    from ucimlrepo import fetch_ucirepo 

    from pathlib import Path

    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder

    from sklearn.pipeline import Pipeline

    from sklearn.impute import SimpleImputer

    import pandas as pd

    import numpy as np

    from scipy import sparse


    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.compose import ColumnTransformer



    # 第一步
    # fetch dataset 
    # 新增输入适配：仅用本地原始数据替代在线读取，下面原清洗语句保持不变。
    fetch_ucirepo = lambda id: dataset
    adult = fetch_ucirepo(id=2) 
  
    # data (as pandas dataframes) 
    X = adult.data.features 
    y = adult.data.targets 
  
    # metadata 
    print(adult.metadata) 
  
    # variable information 
    print(adult.variables)

    # 第二步
    # 查看数据大小
    print("特征数据大小：", X.shape)
    print("目标数据大小：", y.shape)

    # 查看前几行
    print(X.head())

    # 查看各列类型及非空数量
    X.info()

    # 查看 pandas 已识别的缺失值
    print("\n各列缺失值数量：")
    print(X.isna().sum())

    # 查看目标标签，包括缺失值
    print("\n目标标签分布：")
    print(y.iloc[:, 0].value_counts(dropna=False))

    # 检查类别列中是否存在 '?' 或前后空格
    for col in X.select_dtypes(include=["object", "string"]).columns:
        print(f"\n{col} 的取值：")
        print(X[col].unique())

    X_clean = X.copy()
    y_clean = y.iloc[:, 0].copy()

    # 第三步：统一缺失值和标签格式
    # 1. 清理类别变量前后的空格
    categorical_cols = X_clean.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for col in categorical_cols:
        X_clean[col] = (
            X_clean[col]
            .astype("string")
            .str.strip()
            .replace({"?": pd.NA, "": pd.NA})
        )

        # 统一成 sklearn 更容易处理的 object + np.nan
        X_clean[col] = X_clean[col].astype(object)
        X_clean.loc[X_clean[col].isna(), col] = float("nan")

    # 2. 统一目标标签：去掉空格和末尾的句点
    y_clean = (
        y_clean.astype("string")
        .str.strip()
        .str.rstrip(".")
        .replace({"?": pd.NA, "": pd.NA})
    )

    # 3. 检查是否出现预期之外的标签
    unexpected = set(y_clean.dropna().unique()) - {"<=50K", ">50K"}

    if unexpected:
        raise ValueError(f"发现意外的目标标签：{unexpected}")

    # 4. 缺失的目标标签不能用于监督学习，删除对应行
    valid_target = y_clean.notna()
    X_clean = X_clean.loc[valid_target].copy()
    y_clean = y_clean.loc[valid_target].copy()

    # 5. 转换目标标签：低收入为 0，高收入为 1
    y_clean = y_clean.map({"<=50K": 0, ">50K": 1}).astype(int)

    print("清理后的特征大小：", X_clean.shape)
    print("\n清理后的缺失值数量：")
    print(X_clean.isna().sum())
    print("\n清理后的目标分布：")
    print(y_clean.value_counts())

    print("特征完全相同的重复行数：", X_clean.duplicated().sum())

    # 第四步：先划分训练集与测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=0.2,
        random_state=42,
        stratify=y_clean
    )

    # 识别数值列与类别列
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    print("训练集大小：", X_train.shape)
    print("测试集大小：", X_test.shape)
    print("数值列：", numeric_cols)
    print("类别列：", categorical_cols)

    # 方案一：删除缺失样本
    # 删除包含缺失值的样本，并同步筛选标签
    train_complete = X_train.notna().all(axis=1)
    test_complete = X_test.notna().all(axis=1)

    X_train_1 = X_train.loc[train_complete].copy()
    y_train_1 = y_train.loc[train_complete].copy()

    X_test_1 = X_test.loc[test_complete].copy()
    y_test_1 = y_test.loc[test_complete].copy()

    preprocessor_1 = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols
            )
        ]
    )

    X_train_1_processed = preprocessor_1.fit_transform(X_train_1)
    X_test_1_processed = preprocessor_1.transform(X_test_1)

    print("方案一删除的训练样本：", len(X_train) - len(X_train_1))
    print("方案一删除的测试样本：", len(X_test) - len(X_test_1))
    print("处理后的训练集大小：", X_train_1_processed.shape)

    # 方案二：中位数与众数填补

    numeric_pipeline_2 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline_2 = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor_2 = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline_2, numeric_cols),
            ("categorical", categorical_pipeline_2, categorical_cols)
        ]
    )

    X_train_2_processed = preprocessor_2.fit_transform(X_train)
    X_test_2_processed = preprocessor_2.transform(X_test)

    # 样本没有删除，标签保持一致
    y_train_2 = y_train.copy()
    y_test_2 = y_test.copy()

    print("方案二处理后的训练集大小：", X_train_2_processed.shape)
    print("方案二处理后的测试集大小：", X_test_2_processed.shape)

    # 方案三：保留缺失信息
    numeric_pipeline_3 = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True)
            ),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline_3 = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Missing")
            ),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor_3 = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline_3, numeric_cols),
            ("categorical", categorical_pipeline_3, categorical_cols)
        ]
    )

    X_train_3_processed = preprocessor_3.fit_transform(X_train)
    X_test_3_processed = preprocessor_3.transform(X_test)

    y_train_3 = y_train.copy()
    y_test_3 = y_test.copy()

    print("方案三处理后的训练集大小：", X_train_3_processed.shape)
    print("方案三处理后的测试集大小：", X_test_3_processed.shape)

    # 检查方案一、二、三

    def check_processed_data(name, train_data, test_data):
        for split_name, data in [
            ("训练集", train_data),
            ("测试集", test_data)
        ]:
            values = data.data if sparse.issparse(data) else np.asarray(data)

            print(
                f"{name} / {split_name}："
                f"形状={data.shape}，"
                f"所有数值有限={np.isfinite(values).all()}"
            )

        assert train_data.shape[1] == test_data.shape[1], "训练与测试列数不一致"

    check_processed_data(
        "方案一", X_train_1_processed, X_test_1_processed
    )
    check_processed_data(
        "方案二", X_train_2_processed, X_test_2_processed
    )
    check_processed_data(
        "方案三", X_train_3_processed, X_test_3_processed
    )

    # 画图

    save_dir = Path(
        "/Users/andrew/Documents/Linnan University Course/"
        "works_term1/Data Mining/group/11.3/pre_clean_picture"
    )
    # 新增路径适配：保留上一行原路径语句，在运行前绑定当前项目路径。
    save_dir = picture_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # 使用英文标题，避免中文字体缺失导致图片乱码
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })

    def save_picture(fig, filename):
        """保存并显示图片。"""
        fig.savefig(
            save_dir / filename,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )
        plt.show()
        plt.close(fig)
        print("已保存：", save_dir / filename)

    # 准备可视化用的数据副本
    X_vis = X.copy()
    y_vis = y.iloc[:, 0].copy()

    # 仅统一缺失值表示，便于统计
    category_cols = X_vis.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    for col in category_cols:
        X_vis[col] = (
            X_vis[col]
            .astype("string")
            .str.strip()
            .replace({"?": pd.NA, "": pd.NA})
        )

    # 统一标签，避免把 >50K 和 >50K. 统计为两个类别
    y_vis = (
        y_vis.astype("string")
        .str.strip()
        .str.rstrip(".")
        .replace({"?": pd.NA, "": pd.NA})
    )

    numeric_cols = X_vis.select_dtypes(include="number").columns.tolist()

    print("可视化样本数：", len(X_vis))
    print("数值变量：", numeric_cols)
    print("类别变量：", category_cols)

    # 缺失值分布图
    missing_count = X_vis.isna().sum().sort_values(ascending=True)
    missing_pct = missing_count / len(X_vis) * 100

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(missing_count.index, missing_count.values, color="#4C78A8")

    offset = max(float(missing_count.max()) * 0.02, 1)

    for i, (col, count) in enumerate(missing_count.items()):
        ax.text(
            count + offset,
            i,
            f"{count:,} ({missing_pct[col]:.1f}%)",
            va="center",
            fontsize=10
        )

    ax.set_xlim(0, max(float(missing_count.max()) * 1.35, 10))
    ax.set_title("Missing Values Before Cleaning")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Feature")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save_picture(fig, "01_missing_values.png")

    # 收入类别分布图
    income_counts = (
        y_vis.fillna("Missing target")
        .value_counts()
        .reindex(["<=50K", ">50K", "Missing target"], fill_value=0)
    )
    income_counts = income_counts[income_counts > 0]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        income_counts.index,
        income_counts.values,
        color=["#4C78A8", "#F58518", "#999999"][:len(income_counts)]
    )

    for bar, count in zip(bars, income_counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}\n({count / len(y_vis):.1%})",
            ha="center",
            va="bottom"
        )

    ax.set_ylim(0, income_counts.max() * 1.18)
    ax.set_title("Income Class Distribution Before Cleaning")
    ax.set_xlabel("Income")
    ax.set_ylabel("Number of records")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save_picture(fig, "02_income_distribution.png")

    # 数值变量分布图
    ncols = 3
    nrows = int(np.ceil(len(numeric_cols) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 4 * nrows),
        squeeze=False
    )

    for ax, col in zip(axes.flat, numeric_cols):
        sns.histplot(
            data=X_vis,
            x=col,
            bins=40,
            color="#4C78A8",
            ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Number of records")

    for ax in list(axes.flat)[len(numeric_cols):]:
        ax.set_visible(False)

    fig.suptitle("Numeric Feature Distributions Before Cleaning", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_picture(fig, "03_numeric_distributions.png")

    # 数值变量相关性热力图
    correlation = X_vis[numeric_cols].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
        ax=ax
    )

    ax.set_title("Numeric Feature Correlations Before Cleaning")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()
    save_picture(fig, "04_numeric_correlations.png")
    # 新增导出：原脚本只保存在内存中，现在把三套矩阵、标签和处理器返回。
    values = locals()
    return {"cleaned_data": X_clean.assign(income=y_clean), "schemes": {
        n: {"X_train": values[f"X_train_{n}_processed"], "X_test": values[f"X_test_{n}_processed"],
            "y_train": values[f"y_train_{n}"], "y_test": values[f"y_test_{n}"],
            "preprocessor": values[f"preprocessor_{n}"]} for n in (1, 2, 3)}}


def run_preprocessing(args):
    """读取本地原始数据，保存下一模块需要的输入，以及原预处理的展示结果。"""
    import joblib
    import openpyxl  # 原代码要导出 Excel；提前检查依赖，避免其原有 except 静默跳过。
    for name in ("adult.data", "adult.test"):
        if not (args.data_dir / name).is_file():
            raise FileNotFoundError(f"缺少原始数据：{args.data_dir / name}")
    output = new_output(args.work_dir / "preprocessing")
    for number, prepare in enumerate((prepare_scheme1, prepare_scheme2, prepare_scheme3), 1):
        state = prepare(args.data_dir, output)
        if number == 1:
            raw = state.pop("raw")
        # 保存未拟合的处理器，模型仍在每个 CV 训练折内拟合，避免使用全局变换矩阵。
        joblib.dump(state, output / f"scheme{number}.joblib", compress=3)
    dataset = SimpleNamespace(
        data=SimpleNamespace(features=raw.drop(columns="income"), targets=raw[["income"]]),
        metadata={"uci_id": 2, "source": str(args.data_dir), "note": "本地原始文件，未获取在线元数据"},
        variables=raw.dtypes)
    processed = original_preprocessing(dataset, output / "figures")
    joblib.dump(processed, output / "processed_data.joblib", compress=3)
    processed["cleaned_data"].to_csv(output / "cleaned_data.csv", index_label="original_row")
    print(f"预处理完成：{output}")


if __name__ == "__main__":
    run_preprocessing(parse_options("Adult 数据预处理"))

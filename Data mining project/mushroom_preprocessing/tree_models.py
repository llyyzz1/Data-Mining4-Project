from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


PROJECT_FOLDER = Path(__file__).resolve().parent #定义路径
DATA_FILE = (
    PROJECT_FOLDER
    / "output"
    / "data"
    / "mushroom_modeling_base.csv"
)

TARGET_COLUMN = "class"
RANDOM_SEED = 42


def load_mushroom_data():
    """读取预处理阶段生成的主文件，返回特征 X 和目标 y。"""

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到 Mushroom 主文件：{DATA_FILE}\n"
            "请先运行 preprocess_mushroom.py。"
        )

    df = pd.read_csv(DATA_FILE)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"数据中找不到目标列 {TARGET_COLUMN!r}。"
            f"现有字段：{df.columns.tolist()}"
        )

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({
        "e": 0,  # edible
        "p": 1,  # poisonous，作为正类
    })

    if y.isna().any():
        raise ValueError("目标列 class 中有未识别的标签，请检查数据。")

    return X, y


def build_preprocessor(X):
    """建立分类属性的填补与编码流程。"""

    categorical_columns = X.columns.tolist()

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def get_tree_models(X):
    """返回包含预处理步骤的决策树和随机森林 Pipeline。"""

    preprocessor = build_preprocessor(X)

    return {
        "Decision Tree": Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", DecisionTreeClassifier(
                random_state=RANDOM_SEED,
            )),
        ]),

        "Random Forest": Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
    }
def get_tree_param_grids():
    return {
        "Decision Tree": {
            "classifier__max_depth": [None, 5, 10, 20],
            "classifier__min_samples_split": [2, 5, 10],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__max_features": ["sqrt", "log2"],
        },
    }
if __name__ == "__main__":
    X, y = load_mushroom_data()
    models = get_tree_models(X)
    param_grids = get_tree_param_grids()

    print("数据读取成功")
    print("特征大小：", X.shape)
    print("目标类别数量：")
    print(y.value_counts())

    for name, model in models.items():
        print(f"\n模型：{name}")
        print(model.named_steps["classifier"])
        print("调参范围：", param_grids[name]) 
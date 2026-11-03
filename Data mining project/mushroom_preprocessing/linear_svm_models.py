import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


from tree_models import build_preprocessor, load_mushroom_data

def get_linear_models(X):


    preprocessor = build_preprocessor(X)

    return {
        "Logistic Regression": Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(
                l1_ratio=0.0,
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight=None,
                random_state=42,
            )),
        ]),

        "Linear SVM": Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", CalibratedClassifierCV(
                SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight=None,
                    decision_function_shape="ovr",
                    max_iter=-1,
                    random_state=42,
                ),
                method="sigmoid",
                ensemble=False,
            )),
        ]),
    }

def get_linear_param_grids():
    """调参网格。

    sklearn 1.9 中 l1_ratio 是主正则参数：0.0=L2、1.0=L1、其余为 elasticnet。
    liblinear 支持 L1/L2 但不支持 elasticnet；lbfgs 只支持 L2。
    这里统一放在 liblinear 下搜索 L1 与 L2，避免原先 lbfgs 子网格
    与 liblinear-L2 重复训练同一类模型。
    """

    return {
        "Logistic Regression": [
            {
                "classifier__solver": ["liblinear"],
                "classifier__l1_ratio": [0.0, 1.0],
                "classifier__C": [0.01, 1.0, 100.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ],
        "Linear SVM": {
            "classifier__estimator__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__estimator__class_weight": [None, "balanced"],
        },
    }

if __name__ == "__main__":
    X, y = load_mushroom_data()
    models = get_linear_models(X)
    param_grids = get_linear_param_grids()

    print("数据读取成功")
    print("特征大小：", X.shape)
    print("目标类别数量：")
    print(y.value_counts())

    for name, model in models.items():
        print(f"\n模型：{name}")
        print(model.named_steps["classifier"])
        print("调参范围：", param_grids[name])
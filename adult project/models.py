"""模型训练：三个方案分别使用普通函数，参数、十折 CV 和预测语句保持原样。"""

import sys
from itertools import combinations
from math import ceil
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sys.dont_write_bytecode = True
from preprocessing import new_output, parse_options

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "adult"
OUT = ROOT / "outputs" / "models" / "adult_apriori"

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education-num",
           "marital-status", "occupation", "relationship", "race", "sex",
           "capital-gain", "capital-loss", "hours-per-week", "native-country",
           "income"]


def train_scheme1(state):
    """方案一的四种模型；读取预处理状态，不重新读取原始数据。"""
    X_train, X_test = state["X_train"], state["X_test"]
    y_train, preprocess = state["y_train"], state["preprocess"]
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    runs = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        # 新增保存：评估模块直接读取这些预测，不重新训练或重新预测。
        runs.append(dict(name=name, pipe=pipe, pred=pred, prob=prob, cv_scores=cv_scores))
    return runs


def train_scheme2(state):
    """方案二沿用其原有处理器和模型，保留原本只保存 CV 均值的方式。"""
    X_train, X_test = state["X_train"], state["X_test"]
    y_train = state["y_train"]
    # 修复上游整体转为 object 后数值列为空的问题，保留原有填补和编码方式。
    X_train, X_test = X_train.infer_objects(), X_test.infer_objects()
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    runs = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        cv_accuracy = cross_val_score(pipe, X_train, y_train, cv=cv,
                                      scoring="accuracy", n_jobs=-1).mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        runs.append(dict(name=name, pipe=pipe, pred=pred, prob=prob, cv_accuracy=cv_accuracy))
    return runs


def train_scheme3(state):
    """方案三沿用其原有处理器、模型参数和训练方式。"""
    X_train, X_test = state["X_train"], state["X_test"]
    y_train = state["y_train"]
    # 恢复数值列及原方案三的缺失指示、Missing 类别处理，不改变模型参数。
    X_train, X_test = X_train.infer_objects(), X_test.infer_objects()
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = X_train.select_dtypes(exclude="number").columns.tolist()
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
    }
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    runs = []
    for name, model in models.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        cv_accuracy = cross_val_score(pipe, X_train, y_train, cv=cv,
                                      scoring="accuracy", n_jobs=-1).mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        runs.append(dict(name=name, pipe=pipe, pred=pred, prob=prob, cv_accuracy=cv_accuracy))
    return runs


def load_transactions():
    train = pd.read_csv(DATA / "adult.data", names=COLUMNS, skipinitialspace=True, na_values="?")
    test = pd.read_csv(DATA / "adult.test", names=COLUMNS, skiprows=1, skipinitialspace=True, na_values="?")
    df = pd.concat([train, test], ignore_index=True)
    df["income"] = df["income"].astype("string").str.strip().str.rstrip(".")

    # Convert numeric attributes to useful categorical intervals for Apriori.
    df["age"] = pd.cut(df["age"], [0, 25, 45, 65, 200], labels=["young", "middle", "older", "senior"])
    df["hours-per-week"] = pd.cut(df["hours-per-week"], [-1, 35, 45, 60, 200], labels=["low", "normal", "long", "very_long"])
    df["capital-gain"] = df["capital-gain"].fillna(0).gt(0).map({False: "zero", True: "positive"})
    df["capital-loss"] = df["capital-loss"].fillna(0).gt(0).map({False: "zero", True: "positive"})
    df = df.dropna(subset=["income"])
    # pd.cut returns Categorical; convert all columns to object before adding Missing.
    df = df.astype(object).fillna("Missing")

    transactions = []
    for _, row in df.iterrows():
        transactions.append({f"{col}={row[col]}" for col in df.columns})
    return df, transactions


def apriori(transactions, min_support=0.05, max_len=3):
    n = len(transactions)
    minimum = ceil(min_support * n)
    tid_sets = {}
    for tid, transaction in enumerate(transactions):
        for item in transaction:
            tid_sets.setdefault(item, set()).add(tid)
    frequent = { (item,): len(tids) for item, tids in tid_sets.items()
                 if len(tids) >= minimum }
    level = set(frequent)
    k = 2
    while level and k <= max_len:
        candidates = set()
        for candidate in combinations(sorted({item for key in level for item in key}), k):
            if all(candidate[:i] + candidate[i+1:] in level for i in range(k)):
                candidates.add(candidate)
        next_level = {}
        for candidate in candidates:
            tids = tid_sets[candidate[0]].copy()
            for item in candidate[1:]:
                tids &= tid_sets[item]
            count = len(tids)
            if count >= minimum:
                next_level[candidate] = count
        frequent.update(next_level)
        level = set(next_level)
        k += 1
    return frequent


def make_rules(frequent, n, min_confidence=0.60, min_lift=1.0):
    rows = []
    for itemset, count in frequent.items():
        if len(itemset) < 2:
            continue
        support = count / n
        for consequent in itemset:
            antecedent = tuple(x for x in itemset if x != consequent)
            if antecedent not in frequent or (consequent,) not in frequent:
                continue
            confidence = support / (frequent[antecedent] / n)
            lift = confidence / (frequent[(consequent,)] / n)
            if confidence >= min_confidence and lift >= min_lift:
                rows.append({"antecedent": ", ".join(antecedent), "consequent": consequent,
                             "support": support, "confidence": confidence, "lift": lift,
                             "predicts_income": consequent.startswith("income=")})
    return pd.DataFrame(rows).sort_values(["lift", "confidence"], ascending=False) if rows else pd.DataFrame()


def run_apriori(args):
    """沿用附件的函数和参数，仅把输入输出路径接入现有模型入口。"""
    global DATA, OUT
    DATA = args.data_dir
    OUT = args.work_dir / "models" / "adult_apriori"
    OUT.mkdir(parents=True, exist_ok=True)
    df, transactions = load_transactions()
    frequent = apriori(transactions, min_support=0.05, max_len=3)
    rules = make_rules(frequent, len(transactions), min_confidence=0.60, min_lift=1.0)
    itemsets = pd.DataFrame([{"itemset": ", ".join(k), "length": len(k),
                              "support_count": v, "support": v / len(transactions)}
                             for k, v in frequent.items()]).sort_values("support", ascending=False)
    rules.to_csv(OUT / "adult_apriori_rules.csv", index=False)
    itemsets.to_csv(OUT / "adult_apriori_frequent_itemsets.csv", index=False)
    summary = pd.DataFrame([{"dataset": "Adult", "instances": len(df), "min_support": 0.05,
                             "min_confidence": 0.60, "frequent_itemsets": len(itemsets),
                             "rules": len(rules), "income_rules": int(rules["predicts_income"].sum()) if len(rules) else 0,
                             "max_lift": rules["lift"].max() if len(rules) else 0}])
    summary.to_csv(OUT / "adult_apriori_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Results saved to:", OUT)


def run_models(args):
    """三个输入文件对应三个方案；每个输出含测试标签和四组模型结果。"""
    import joblib
    prepared = args.work_dir / "preprocessing"
    for number in (1, 2, 3):
        if not (prepared / f"scheme{number}.joblib").is_file():
            raise FileNotFoundError(f"缺少预处理结果：{prepared / f'scheme{number}.joblib'}；请先运行 preprocessing.py")
    output = new_output(args.work_dir / "models")
    for number, train in enumerate((train_scheme1, train_scheme2, train_scheme3), 1):
        print(f"开始训练方案 {number}：四种模型，每种保留原来的十折交叉验证。", flush=True)
        state = joblib.load(prepared / f"scheme{number}.joblib")
        saved = {"y_test": state["y_test"], "runs": train(state)}
        joblib.dump(saved, output / f"scheme{number}.joblib", compress=3)
    run_apriori(args)
    print(f"模型训练完成：{output}")


if __name__ == "__main__":
    run_models(parse_options("Adult 模型训练"))
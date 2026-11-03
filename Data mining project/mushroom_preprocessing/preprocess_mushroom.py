from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

project_folder = Path(__file__).resolve().parent # "设置路径"

source_file = project_folder/ "data"/ "agaricus-lepiota.data"

output_folder = project_folder / "output"
data_folder = output_folder / "data"
tables_folder = output_folder / "tables"
figures_folder = output_folder / "figures"

data_folder.mkdir(parents=True, exist_ok=True)
tables_folder.mkdir(parents=True, exist_ok=True)
figures_folder.mkdir(parents=True, exist_ok=True)
#字段
columns = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]

if not source_file.exists():
    raise FileNotFoundError(
        "找不到原始数据文件：\n"
        f"{source_file}\n\n"
        "请确认 agaricus-lepiota.data 位于 data 文件夹中。"
    )

# header=None：原始文件没有字段名
# names=columns：添加字段名
# na_values=["?"]：把 ? 识别为缺失值
# dtype="string"：全部属性按照分类文本读取

df = pd.read_csv(
    source_file,
    header=None,
    names=columns,
    na_values=["?"],
    keep_default_na=True,
    dtype="string",
)
#结构检查
expected_columns = len(columns)
if df.shape[1] != expected_columns:
    raise ValueError(
        f"应该有 {expected_columns} 列，"
        f"但实际读取到 {df.shape[1]} 列。"
    )
print("=" * 60)
print("MUSHROOM DATA AUDIT")
print("=" * 60)
print(f"实例数量：{len(df)}")
print(f"总列数（包含目标列）：{df.shape[1]}")
print(f"输入属性数量：{df.shape[1] - 1}")
print(f"重复行数量：{df.duplicated().sum()}")
#类别分布
class_name_map = {
    "e": "edible",
    "p": "poisonous",
}

class_distribution = (
    df["class"]
    .value_counts(dropna=False)
    .rename_axis("class_code")
    .reset_index(name="count")
)

class_distribution["class_name"] = (
    class_distribution["class_code"].map(class_name_map)
)

class_distribution["percent"] = (
    class_distribution["count"] / len(df) * 100
).round(2)

print("\n类别分布：")
print(class_distribution.to_string(index=False))
#缺失值
missing_summary = pd.DataFrame({
    "attribute": df.columns,
    "missing_count": [
        int(df[column].isna().sum())
        for column in df.columns
    ],
    "missing_percent": [
        round(df[column].isna().mean() * 100, 2)
        for column in df.columns
    ],
})

missing_summary = missing_summary.sort_values(
    by="missing_count",
    ascending=False,
)

print("\n存在缺失值的属性：")

attributes_with_missing = missing_summary[
    missing_summary["missing_count"] > 0
]

if attributes_with_missing.empty:
    print("没有缺失值")
else:
    print(attributes_with_missing.to_string(index=False))

# 常量属性只有一个非缺失取值，没有分类能力
constant_columns = [
    column
    for column in df.columns
    if column != "class"
    and df[column].nunique(dropna=True) <= 1
]

print("\n常量属性：")

if constant_columns:
    print(constant_columns)
else:
    print("没有常量属性")
#重复
duplicate_count = int(df.duplicated().sum())

print(f"\n重复行数量：{duplicate_count}")

# 本数据集没有重复行。
# 如果以后出现重复行，这里生成删除重复后的审计版本。
df_without_duplicates = df.drop_duplicates().copy()
#生成schema
schema_rows = []

for column in df.columns:
    value_counts = df[column].value_counts(dropna=True)

    if value_counts.empty:
        mode_value = None
        mode_count = 0
    else:
        mode_value = value_counts.index[0]
        mode_count = int(value_counts.iloc[0])

    schema_rows.append({
        "attribute": column,
        "role": (
            "target"
            if column == "class"
            else "feature"
        ),
        "type": "nominal",
        "unique_non_missing": int(
            df[column].nunique(dropna=True)
        ),
        "missing_count": int(
            df[column].isna().sum()
        ),
        "missing_percent": round(
            df[column].isna().mean() * 100,
            2,
        ),
        "mode": mode_value,
        "mode_count": mode_count,
        "is_constant": column in constant_columns,
    })

schema = pd.DataFrame(schema_rows)

# 保留原始审计数据
audit_data = df_without_duplicates.copy()

# 删除常量属性 veil-type
# 功能上对应 Weka RemoveUseless
modeling_base = audit_data.drop(
    columns=constant_columns
).copy()

print("\n删除的无用常量属性：")
print(constant_columns)

print(
    "删除常量属性后的输入属性数量：",
    modeling_base.shape[1] - 1,
)
# 方案 A：
# 保留 stalk-root 缺失值。
# 后续由建模人员在交叉验证 Pipeline 内进行众数填补。
imputation_base = modeling_base.copy()

# 方案 B：
# 删除包含缺失值的行。
complete_cases = modeling_base.dropna().copy()

# 方案 C：
# 删除 stalk-root 整列，保留所有实例。
without_stalk_root = modeling_base.drop(
    columns=["stalk-root"]
).copy()

print("\n缺失值处理方案：")
print(
    "方案 A：保留缺失值，行数 =",
    len(imputation_base),
)
print(
    "方案 B：删除缺失行，行数 =",
    len(complete_cases),
)
print(
    "方案 C：删除 stalk-root，行数 =",
    len(without_stalk_root),
)

# 原始审计版本：
# 保留 veil-type 和 stalk-root 缺失值
audit_data.to_csv(
    data_folder / "mushroom_audit_with_missing.csv",
    index=False,
    na_rep="?",
)

# 推荐建模主文件：
# 删除 veil-type，但保留 stalk-root 缺失值
imputation_base.to_csv(
    data_folder / "mushroom_modeling_base.csv",
    index=False,
    na_rep="?",
)

# 对照方案：删除缺失行
complete_cases.to_csv(
    data_folder / "mushroom_complete_cases.csv",
    index=False,
)

# 对照方案：删除 stalk-root
without_stalk_root.to_csv(
    data_folder / "mushroom_without_stalk_root.csv",
    index=False,
)

schema.to_csv(
    tables_folder / "schema_and_missingness.csv",
    index=False,
)

class_distribution.to_csv(
    tables_folder / "class_distribution.csv",
    index=False,
)

missing_summary.to_csv(
    tables_folder / "missing_values_summary.csv",
    index=False,
)

constant_summary = pd.DataFrame({
    "constant_attribute": constant_columns
})

constant_summary.to_csv(
    tables_folder / "constant_attributes.csv",
    index=False,
)


audit_summary = pd.DataFrame([
    {
        "instances": len(df),
        "features_before_cleaning": df.shape[1] - 1,
        "features_after_removing_constants":
            modeling_base.shape[1] - 1,
        "duplicate_rows": duplicate_count,
        "total_missing_values":
            int(df.isna().sum().sum()),
        "stalk_root_missing":
            int(df["stalk-root"].isna().sum()),
        "stalk_root_missing_percent":
            round(
                df["stalk-root"].isna().mean() * 100,
                2,
            ),
        "complete_case_instances":
            len(complete_cases),
        "edible_instances":
            int((df["class"] == "e").sum()),
        "poisonous_instances":
            int((df["class"] == "p").sum()),
        "constant_attributes":
            ", ".join(constant_columns),
    }
])

audit_summary.to_csv(
    tables_folder / "audit_summary.csv",
    index=False,
)



plot_class_distribution = (
    df["class"]
    .map(class_name_map)
    .value_counts()
    .reindex(["edible", "poisonous"])
)

plt.figure(figsize=(7, 5))

bars = plt.bar(
    plot_class_distribution.index,
    plot_class_distribution.values,
    color=["#287271", "#C8553D"],
)

plt.title("Mushroom Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Instances")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        str(int(bar.get_height())),
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    figures_folder / "class_distribution.png",
    dpi=200,
)

plt.close()


plot_missing_values = df.isna().sum()
plot_missing_values = plot_missing_values[
    plot_missing_values > 0
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    plot_missing_values.index,
    plot_missing_values.values,
    color="#4C78A8",
)

plt.title("Missing Values by Attribute")
plt.xlabel("Attribute")
plt.ylabel("Number of Missing Values")

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        str(int(bar.get_height())),
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    figures_folder / "missing_values.png",
    dpi=200,
)

plt.close()

odor_class_table = pd.crosstab(
    df["odor"],
    df["class"],
)

odor_class_table = odor_class_table.rename(
    columns=class_name_map
)

odor_class_table.plot(
    kind="bar",
    stacked=True,
    figsize=(10, 6),
    color=["#287271", "#C8553D"],
)

plt.title("Mushroom Class Distribution by Odor")
plt.xlabel("Odor Code")
plt.ylabel("Number of Instances")
plt.legend(title="Class")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    figures_folder / "class_by_odor.png",
    dpi=200,
)

plt.close()

print("\n" + "=" * 60)
print("预处理完成")
print("=" * 60)

print(f"原始实例数：{len(df)}")
print("原始输入属性数：22")
print(f"重复行：{duplicate_count}")
print(
    "stalk-root 缺失值：",
    int(df["stalk-root"].isna().sum()),
)
print(
    "stalk-root 缺失率：",
    f"{df['stalk-root'].isna().mean() * 100:.2f}%",
)
print(f"常量属性：{constant_columns}")
print(f"完整案例数量：{len(complete_cases)}")
print(f"Edible：{int((df['class'] == 'e').sum())}")
print(
    f"Poisonous：{int((df['class'] == 'p').sum())}"
)

print("\n推荐交给建模成员的主文件：")
print(
    data_folder / "mushroom_modeling_base.csv"
)

print("\n所有输出文件位于：")
print(output_folder)
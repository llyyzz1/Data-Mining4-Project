# Adult 与 Mushroom：统一基线补充实验

本目录新增两项内容：为 Mushroom 补充训练集内的十折交叉验证；使用与 Adult 相同的四种分类器参数，生成跨数据集比较表和图。两个原项目的代码、数据和旧结果均保留。

## 运行

在 PowerShell 中执行（不要求切换当前目录）：

```powershell
& 'G:/anaconda/python.exe' -X utf8 -B 'G:/GPTLEARNING/cross_dataset_comparison/compare_baselines.py'
```

脚本默认读取同级的 `adult project` 和 `Data mining project/mushroom_preprocessing`。也可以通过 `--adult-dir`、`--mushroom-dir`、`--output-dir` 指定路径；相对路径均以脚本所在目录为基准。`--cv-jobs 2` 控制并行验证折数。再次运行会更新本目录的实验结果。

需要两个原项目现有的预处理结果，以及 Adult 方案二保存的模型和正式评估表。缺失输入或训练失败会直接报错；不会跳过失败模型。

## 两项补充的具体做法

1. **Mushroom 十折评估**：将数据按类别分层划分为 80% 训练集和 20% 独立测试集，随机种子为 42。仅在训练集中执行 `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`；每折重新拟合包含预处理的完整 Pipeline。报告十折 Accuracy 和完整训练集拟合后的独立测试指标。原 `model_tuning.py` 的五折网格搜索继续作为调参实验保留，本次没有重跑网格搜索。
2. **统一分类器参数**：固定 Adult 方案二作为比较参考，读取现有保存结果；逐项检查四个分类器的全部参数，再用这些参数重新训练 Mushroom。不会从 Adult 的三个方案中按模型挑选最好成绩，也不重新训练 Adult。

| 分类器 | 两个数据集共同配置 |
| --- | --- |
| Decision Tree | `max_depth=8, random_state=42` |
| Random Forest | `n_estimators=200, random_state=42, n_jobs=-1` |
| Logistic Regression | `max_iter=1000, random_state=42`，其余保持当前版本默认值 |
| SVM | `SVC(probability=True, random_state=42)`，默认 RBF 核 |

未列出的全部参数记录在 `outputs/experiment_config.json`。Mushroom 原实验的线性 SVM 与本次 RBF SVM 是不同配置；报告应明确写成两组实验，不能把原调参成绩作为本次统一基线成绩。

## 输出文件

所有新结果写入 `outputs/`：

- `cross_dataset_metrics.csv`：两个数据集 × 四种模型，共八行；包含 CV 均值和测试集 Accuracy、Precision、Recall、F1、ROC-AUC、Kappa。
- `cross_dataset_metrics.png`：测试集 Accuracy、F1、Recall、ROC-AUC 对比。
- `cross_dataset_cv_accuracy.png`：训练集十折 Accuracy 均值对比。
- `adult_scheme2_reference.csv`：从 Adult 保存预测重新计算并与原评估表核对的参考成绩。
- `experiment_config.json`：参数、协议、实际运行的库版本及说明。
- `source_preservation.json`：运行前后原项目文件 SHA-256 核对记录。
- `run_manifest.json`：只有 `status=complete` 表示完整运行成功。
- `mushroom/cv_scores_10fold.csv`：四种模型各十折，共 40 个验证分数。
- `mushroom/metrics_10fold_holdout.csv`：Mushroom 的 CV 均值和独立测试指标。
- `mushroom/split_and_cv_assignments.csv`：每条记录的训练/测试归属和验证折号。
- `mushroom/test_predictions.csv`：四种模型在测试集上的实际标签、预测标签和正类概率。
- `mushroom/models.joblib`：拟合后的 Pipeline、测试行号、预测、概率及逐折分数。
- `mushroom/confusion_matrices.csv`、`mushroom/confusion_matrices.png`：四种模型的测试混淆矩阵。

Adult 方案二原文件只保存 CV 均值，因此不能从中恢复十个单折分数或标准差；比较图只展示双方的均值。

## 解读时需要说明的限制

- **预处理保持各自方案**。Adult 方案二对数值特征填补、标准化，对分类特征填补、独热编码。Mushroom 直接复用原 `tree_models.py` 中的数据读取和预处理函数，全为分类特征。统一的是分类器参数和评估协议，不能把成绩差异全部归因于数据集本身。
- Mushroom 共 8,124 条，训练 6,499 条、测试 1,625 条；移除常量列后的输入有 21 个特征。原读取方式把 2,480 个 `?` 当作普通类别，不是 NaN；原 `SimpleImputer` 不会填补这些问号，本次继续保留该处理。不要在报告中描述成已用众数填补了问号。
- 正类定义不同：Adult 为收入 `>50K`，Mushroom 为有毒 `poisonous`。类别比例和任务难度不同，需要结合 Precision、Recall、F1、AUC 解释，不能仅按 Accuracy 判断哪个任务或模型更好。
- Adult 沿用原结果，其原始重复记录风险仍需在报告中说明；本次补充实验没有改变原数据清洗方式。
- Mushroom 接近或达到满分表示当前划分上的成绩，不代表真实场景中不会误判。跨数据集讨论还应结合特征类型、样本量、类别平衡和可分性。

## 依赖与复现

运行依赖 `numpy`、`pandas`、`scikit-learn`、`joblib` 和 `matplotlib`。本次使用现有 Anaconda 环境，不安装或替换原项目的依赖；实际版本见配置文件。保存的模型建议在相同版本环境中加载。

当前 scikit-learn 1.9.0 对 `SVC(probability=True)` 会产生弃用警告，当前版本仍可执行。本次保留该配置以与 Adult 一致，未屏蔽警告，也未改用另一种校准模型。

本次文件适合作为报告和展示的补充实验材料；仍需在最终报告中解释结果、填写成员贡献，并按课程要求准备演示。

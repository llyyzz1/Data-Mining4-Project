# Adult：Evaluation 演示内容与讲稿

准备日期：2026-10-26。依据当前项目 `outputs/` 中的正式结果整理。

本文件只提供展示材料，不修改任何 Python 代码、模型参数、预测或结果。建议制作 4 页 PPT，讲解约 2 分 40 秒；现场运行、Apriori 和问答可作为备用内容。这里没有将全组要求的 Weka 演示替换为 Python 演示，Weka 部分仍需由全组安排或向老师确认。

## 一、可直接复制到 PPT 的内容

### 第 1 页：Evaluation Workflow（约 30 秒）

**页面内容**

- 任务：预测个人年收入是否超过 50K；正类 `1 = >50K`，负类 `0 = <=50K`。
- 比较范围：3 个预处理方案 × 4 个分类器，共 12 组模型结果。
- 训练阶段：训练集内进行十折分层交叉验证；填补、编码与标准化放在 Pipeline 内。
- 测试阶段：模型在训练集拟合后，对留出测试集预测；方案二、三为 80%/20% 分层划分。
- Evaluation 模块读取保存的标签、预测和概率，计算指标、生成图表，不重新训练。

**讲稿**

> 我负责模型评估与结果展示。我们的目标是判断一个人的年收入是否超过五万美元，正类表示超过五万美元。我们比较三个预处理方案和四种分类器，共十二组结果。训练阶段使用十折分层交叉验证，并在每一折内完成预处理；之后再在留出测试集上评估。我的模块读取已经保存的预测和概率，不会重新训练模型。

**展示位置**

- 输入：`G:/GPTLEARNING/adult_project_three_files_simple/outputs/models/scheme1.joblib`、`scheme2.joblib`、`scheme3.joblib`。
- 入口：`G:/GPTLEARNING/adult_project_three_files_simple/evaluation.py` 中的 `run_evaluation()`。
- 方案一删除了缺失记录，最终训练集 36,183 条、测试集 9,039 条；方案二、三分别为 39,073 条、9,769 条。

### 第 2 页：Why Multiple Metrics?（约 30 秒）

**页面内容**

| 指标 | 演示时的解释 |
|---|---|
| Accuracy | 所有测试样本中预测正确的比例 |
| Precision | 被预测为高收入的人中，实际高收入的比例 |
| Recall | 实际高收入的人中，被模型识别出来的比例 |
| F1 | Precision 和 Recall 的调和平均 |
| ROC-AUC | 衡量不同阈值下模型区分高、低收入的排序能力 |
| Kappa | 补充展示：扣除随机一致后的分类一致程度 |

- 全量数据的高收入样本约占 23.93%，存在类别不平衡。
- 仅看 Accuracy 可能忽略大量漏判，因此同时看 Recall、F1 和 ROC-AUC。
- 表中的 Precision、Recall、F1 都以 `>50K` 为正类计算，不是宏平均或加权平均。

**讲稿**

> 高收入样本只占约百分之二十四，所以不能只看准确率。Precision 说明预测为高收入的人有多少是真的，Recall 说明真正的高收入者有多少被识别出来。F1 综合这两项，ROC-AUC 则观察不同阈值下的区分能力。我们同时展示这些指标，避免因为多数类预测得好就认为模型全面优秀。

### 第 3 页：Results and Model Comparison（约 45 秒）

**页面内容**

| 当前测试集观察 | 方案与模型 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Accuracy 最高 | 方案三 SVM | 86.22% | 77.56% | 59.71% | 0.6747 | 0.8984 |
| F1 最高 | 方案二随机森林 | 85.63% | 73.19% | 63.05% | 0.6774 | 0.9054 |
| ROC-AUC 最高 | 方案二决策树 | 85.79% | 79.84% | 54.36% | 0.6468 | 0.9066 |

- 不同指标的领先模型不同，选择依据应与任务目标一致。
- 方案二、三使用相同测试样本，方案三相对方案二的效果依模型而变化。
- 跨三个方案的总体排名是描述性结果：方案一的测试人群与另两方案不同。
- 这些小幅差异尚未做显著性检验，不宣称具有统计显著优势。

**推荐配图**

![三个预处理方案的 F1 比较](G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/adult_comparison/f1_comparison.png)

如 PPT 空间不足，使用配图加两条结论：`Highest Accuracy: Scheme 3 SVM (86.22%)`；`Highest F1: Scheme 2 Random Forest (0.6774)`。完整表放备份页。

**讲稿**

> 结果中，方案三的 SVM 测试准确率最高，约百分之八十六点二二；方案二的随机森林 F1 最高，为零点六七七四。方案二决策树的 AUC 最高，但 Recall 较低，说明概率排序表现好，并不等于当前分类阈值下能识别最多高收入者。所以我们不能简单宣布某个模型在所有方面都最好。方案之间的差异也比较小，目前不宣称有统计显著优势。

### 第 4 页：Confusion Matrix and Limitations（约 45 秒）

**页面内容**

以方案二随机森林为例，测试集共 9,769 条：

| 实际类别 / 预测类别 | 预测 <=50K（0） | 预测 >50K（1） |
|---|---:|---:|
| 实际 <=50K（0） | 6,891（TN） | 540（FP） |
| 实际 >50K（1） | 864（FN） | 1,474（TP） |

- 识别了 1,474 名高收入者，漏判 864 名；Recall = 1,474 / 2,338 = 63.05%。
- 540 名低收入者被误判为高收入；Precision = 1,474 / 2,014 = 73.19%。
- 限制：方案一删除缺失记录后测试样本不同；数据存在重复记录跨集合重叠风险。
- 后续：补充共同测试样本上的预处理比较；调查重复记录；以交叉验证支持模型选择。

**推荐配图**

![方案二四种模型的混淆矩阵](G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/scheme2/scheme2_confusion_matrices.png)

截图时可保留整张图并框选右上角 Random Forest 子图。横轴为预测，纵轴为实际，`0 = <=50K`，`1 = >50K`。

**讲稿**

> 以方案二随机森林为例，模型正确识别了一千四百七十四名高收入者，但仍漏掉八百六十四名，Recall 约百分之六十三；同时有五百四十名低收入者被误判。这说明即使准确率超过百分之八十五，漏判问题仍然存在。最后，我们也说明两个限制：方案一删除缺失样本后，测试人群不同；重复记录可能跨训练和测试集重叠。因此结果需要结合这些限制解释，不能只依据测试排名确定最终模型。

## 二、现场运行与展示顺序

### 演示前准备

1. 确认 `G:/GPTLEARNING/adult_project_three_files_simple/outputs/models/` 下有三个 `scheme*.joblib` 文件。
2. 使用已经安装依赖的 `G:/anaconda/python.exe`。
3. 提前打开下面列出的结果图和汇总表，作为现场运行失败时的备用展示。
4. 下述命令会覆盖同名评估输出；本文件整理过程中没有执行它们，也没有重新训练模型。

### 建议现场只运行 Evaluation

```powershell
& 'G:/anaconda/python.exe' 'G:/GPTLEARNING/adult_project_three_files_simple/evaluation.py'
```

该命令基于文件位置读取默认路径，可以从其他目录运行。它读取已有模型输出，生成指标和图表。若显示缺少模型结果，应先完成模型阶段；不要把缺输入解释为评估算法失败。

**可照读的操作说明**

> 这里我只运行 evaluation.py。它读取模型成员已经保存的三个方案结果，计算指标、生成 ROC 曲线、混淆矩阵和模型比较图，最后输出汇总表。这个步骤不会重复训练模型。

**运行后按以下顺序打开**

| 顺序 | 文件绝对路径 | 说明 |
|---|---|---|
| 1 | `G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/adult_comparison/adult_all_schemes_results.xlsx` | 说明三方案、四模型，共十二行结果 |
| 2 | `G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/adult_comparison/f1_comparison.png` | 解释指标不同，排名不同 |
| 3 | `G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/scheme2/scheme2_confusion_matrices.png` | 讲清楚漏判和误判 |
| 4（备用） | `G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/scheme2/scheme2_roc.png` | 解释 ROC 横轴 FPR、纵轴 TPR；AUC 反映排序能力 |

不要逐个打开全部 16 张图。决策树图可交给模型成员解释，你的核心是评估协议、指标和结果含义。

### 如老师要求展示完整入口

```powershell
& 'G:/anaconda/python.exe' 'G:/GPTLEARNING/adult_project_three_files_simple/evaluation.py' --all
```

该模式依次执行预处理、模型训练和评估。模型阶段包含 12 组模型及十折 CV，全量训练不适合临时放进 2–3 分钟的个人讲解。可事先完成运行，现场展示入口及保存的结果。

## 三、Apriori 备用展示（约 20–30 秒，不计入四页主讲稿）

Apriori 在 `models.py` 中运行，其结果位于 `outputs/models/adult_apriori/`。当前 `evaluation.py` 不读取或绘制这些规则，因此现场只运行评估入口不会重新生成 Apriori 结果。若模型成员已讲解，可省略这一段。

**已有结果**

- 全量 48,842 条记录；最小支持度 0.05，最小置信度 0.60，最小提升度 1.0；最大项集长度 3。
- 1,886 个频繁项集，1,864 条规则，其中 208 条以收入为后件。
- 示例：`marital-status=Married-civ-spouse, occupation=Exec-managerial → income=>50K`。
- 示例指标：支持度约 5.02%，置信度约 68.08%，提升度约 2.85。

**解释**

> 支持度表示同时满足前件和后件的记录占全体的比例；置信度表示满足前件时后件出现的比例；提升度比较这一比例与总体高收入比例。该规则显示关联，不说明婚姻或职业导致高收入，也不是留出测试集上的预测准确率。

文件：`G:/GPTLEARNING/adult_project_three_files_simple/outputs/models/adult_apriori/adult_apriori_rules.csv`。

## 四、常见问题与回答

### 1. 十折交叉验证是在 evaluation.py 里做的吗？

不是。它在 `models.py` 中对训练集执行；`evaluation.py` 读取其结果和测试预测。CV 的预处理位于 Pipeline 内，避免在分折前用整份训练数据拟合预处理器。

### 2. CV Accuracy 和 Accuracy 有什么区别？

`CV Accuracy (mean)` 是训练集十折验证准确率的平均值；`Accuracy` 是最后模型在留出测试集上的准确率。它们不是同一批样本、同一个评估过程。接近不等于证明没有过拟合。

### 3. 哪个模型最好？

当前测试观察中，方案三 SVM 的 Accuracy 最高，方案二随机森林的 F1 最高，方案二决策树的 ROC-AUC 最高。应先确定评价目标，再利用训练集交叉验证支持选择；不根据测试集多次挑选、调整模型。

### 4. 为什么方案三不是始终更好？

保留缺失信息不保证对每种模型都有收益。方案二、三测试样本一致，实际结果中 SVM 和逻辑回归的部分指标提高，随机森林的部分指标下降。当前数据的六个数值列没有缺失，所以数值缺失指示器在本次数据上没有新增有效信息；主要差异来自类别缺失处理。不要声称数值缺失指示器造成了性能提升。

### 5. 为什么方案一不能直接与另两方案作严格比较？

方案一删除了有缺失值的测试记录，留下 9,039 条；方案二、三有 9,769 条。差异可能来自测试人群变化，也可能来自预处理。共同测试样本比较尚未加入正式输出，这是限制，而不是已经完成的补充实验。

### 6. 数据是否存在泄漏风险？

学习型预处理在 CV 的 Pipeline 内完成；但仍发现重复记录跨训练和测试集重叠的风险，需要调查并讨论。不能仅因为使用了 Pipeline，就宣布不存在任何泄漏风险。Apriori 对全量数据作描述性关联分析；若利用其规则再改造监督模型，则应只在训练数据内挖掘规则。

### 7. 哪些改进已经做了，哪些只是建议？

已完成：三方案、四分类器、十折 CV、留出评估、十二组正式指标、十六张正式图、Apriori 结果。尚未完成：共同测试集补充实验、重复记录治理实验、显著性检验，以及基于当前结果的完整报告。不要在演示中把建议说成已完成。

### 8. 需要展示第二个数据集吗？

你负责讲 Adult 的评估部分。整组仍需由另一数据集成员和统筹成员补上跨数据集比较；这不是当前文件夹已经完成的内容。

## 五、数据来源与口径

- 正式指标：`G:/GPTLEARNING/adult_project_three_files_simple/outputs/evaluation/adult_comparison/adult_all_schemes_results.csv`。
- 测试标签、预测、概率和训练管道：`G:/GPTLEARNING/adult_project_three_files_simple/outputs/models/scheme1.joblib`、`scheme2.joblib`、`scheme3.joblib`。
- Apriori 汇总与规则：`G:/GPTLEARNING/adult_project_three_files_simple/outputs/models/adult_apriori/`。
- 不使用 `verification/` 中的 240 行采样测试指标作为正式实验成绩。
- PPT 中百分比保留两位小数，F1/AUC 保留四位小数；源 CSV 保留原精度。
- “最高”表示当前已运行设置下的观察值，不代表已证明统计显著，也不代表对其他数据集普遍最优。

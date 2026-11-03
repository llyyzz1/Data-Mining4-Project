# Adult 与 Mushroom 数据挖掘项目

本目录包含本次作业实际使用的 10 份 Python 代码、两个 UCI 原始数据集、
Adult 已训练模型、预处理结果、评估表格、图表、跨数据集比较、现有报告和
Adult 演示材料。全部原代码逐字节复制，未改变算法、参数、注释和运行方式。

## 为什么保留英文空格目录名

跨数据集脚本默认查找同级 `adult project` 和
`Data mining project/mushroom_preprocessing`。保留原名可以直接使用这些
路径，而无需改写代码。Adult 仍只有 preprocessing.py、models.py、evaluation.py
三个 Python 文件；本次没有新增 Python 入口或适配器。

## 安装

已核验的环境为 Python 3.14.6，包版本见根目录 requirements.txt。
如需新环境，在根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

两个原项目内的 requirements.txt 均原样保留，其历史版本不同。
整合后请优先使用根目录这一份，不要把两个旧版本要求同时安装。
本次核验使用已有 Anaconda 环境，没有另建环境并在线安装全部依赖。

## 运行

以下命令都从本目录执行，路径需保留引号。

Adult 三个独立阶段：

```powershell
python -X utf8 -B "adult project/preprocessing.py"
python -X utf8 -B "adult project/models.py"
python -X utf8 -B "adult project/evaluation.py"
```

Adult 完整流程：

```powershell
python -X utf8 -B "adult project/evaluation.py" --all
```

目录已附训练好的 Adult 模型，第三条评估命令可以直接运行，不需要重新训练。
完整 Adult 训练包含三个方案、每个四种模型和十折验证，耗时较长。
重复执行会更新同名输出。需要另存 Adult 输出时可加
`--work-dir outputs/new_run`，但比较脚本默认仍读取标准 outputs 目录。

Mushroom 预处理：

```powershell
python -X utf8 -B "Data mining project/mushroom_preprocessing/preprocess_mushroom.py"
```

Mushroom 原建模、评估、五折网格搜索和 Apriori：

```powershell
python -X utf8 -B "Data mining project/mushroom_preprocessing/evaluate_models.py"
```

该评估文件内部会训练模型，并根据保留的两个 True 开关继续调参和运行 Apriori，
与 Adult 仅读取已保存预测的评估方式不同。tree_models.py 和
linear_svm_models.py 直接运行只打印模型配置，不等于完成训练。
model_tuning.py 与 association_rules.py 也可分别运行。

跨数据集的统一参数十折验证与独立测试：

```powershell
python -X utf8 -B "cross_dataset_comparison/compare_baselines.py" --cv-jobs 2
```

比较需要的前置文件已附齐。若从原始数据重新计算，先运行 Adult 的 --all，
再运行 Mushroom 预处理，最后运行比较脚本。原 Mushroom 五折调参是另一组
实验，不能当作统一十折基线。两者预处理保持各自方法，并非全部数值配置一致。

## 重点文件

- 跨数据集结果：cross_dataset_comparison/outputs/cross_dataset_metrics.csv
- 两数据集比较图：同目录 cross_dataset_metrics.png
- Adult 三方案成绩：adult project/outputs/evaluation/adult_comparison/
- Mushroom 原结果：Data mining project/mushroom_preprocessing/output/tables/
- 原图：对应结果目录下的 PNG
- 中英文现有 Word 报告：reports/
- 所有文件校验：CHECKSUMS.sha256、FILE_MANIFEST.json
- 原代码保护检查：CODE_PRESERVATION.json
- 上传教程：docs/GITHUB_UPLOAD_zh.md
- 本次实际验证范围：docs/VALIDATION.md

原项目 README 中有已经过时的文件夹名、未修复说明和早期测试状态，本次保留原文。
实际使用以根目录 README 和当前代码为准。现有 Word 中仍可能有学号、团队名、
旧贡献比例等待确认信息；本次整理未改写报告，也没有附入更早的旧 PDF 冒充最新版。
presentation 内是已有 Adult 评估演示，不是完整六人跨数据集演示。

## 上传

三份 Adult 模型分别约 34–36 MiB，无法通过 GitHub 网页直接拖拽上传。
所有单文件都小于 Git 推送的 100 MiB 限制，请按照 docs/GITHUB_UPLOAD_zh.md
使用 Git 上传整个目录。外层 ZIP 只是方便搬运，不能代替仓库中的展开代码。

已排除旧版本、重复运行输出、旧测试副本、缓存及其他无关项目。
原目录没有被修改；本次没有联网推送，也没有创建 GitHub 仓库。

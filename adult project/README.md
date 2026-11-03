# Adult 三文件精简版

本版仅包含 preprocessing.py、models.py、evaluation.py 三个 Python 文件。
旧版三文件和 G:/adult final 中的五份原文件均未改动。

## 本次精简

- 删除运行时 AST 解析、exec 执行器和多行源码字符串，改用普通函数。
- 删除运行时哈希、版本检查、阶段清单和通用配置类。
- 保留必要的路径参数、joblib 保存/读取、输入缺失提示、输出目录保护和 --all。
- 给新增代码及每个函数增加中文说明，标明其职责和路径适配原因。
- 原算法语句、变量名、模型参数、原有注释均保留；结构调整仅为函数包裹、缩进和阶段拆分。

preprocessing.py 仍较长，因为其中包含原 pre_clean.py 的完整清洗/绘图，以及原三个模型脚本各自的数据准备。
这些属于原有业务代码，不是新的运行框架。模型和评估语句已分别放入对应模块，可以直接阅读和设置断点。

## 文件和职责

```text
adult_project_three_files_simple/
├── preprocessing.py       原始数据读取、三个方案准备、原清洗/探索图、结果导出
├── models.py              原四种模型、十折 CV、拟合、预测、模型结果保存
├── evaluation.py          原指标、ROC/混淆矩阵/树图、方案比较、--all
├── adult/                 原始 Adult 数据和说明，逐字节复制
├── requirements.txt       本机依赖版本
├── README.md
├── verification/          检查报告和日志，不属于正式模型成绩
└── outputs/               正式运行后生成
    ├── preprocessing/
    │   ├── scheme1.joblib、scheme2.joblib、scheme3.joblib
    │   ├── processed_data.joblib
    │   ├── cleaned_data.csv
    │   └── figures/       原四张探索图
    ├── models/
    │   └── scheme1.joblib、scheme2.joblib、scheme3.joblib
    └── evaluation/
        ├── adult_scheme1/ 原方案一的指标 CSV/Excel 和三张图
        ├── scheme2/       原方案二的指标 CSV/Excel 和三张图
        ├── scheme3/       原方案三的指标 CSV/Excel 和三张图
        └── adult_comparison/ 汇总 CSV/Excel 和三个比较图
```

## 运行

本机使用 G:/anaconda/python.exe，依赖已安装。依赖版本列在 requirements.txt，验证环境为 Python 3.14.6。

```powershell
cd G:\GPTLEARNING\adult_project_three_files_simple
python preprocessing.py
python models.py
python evaluation.py
```

或者从头执行全部步骤：

```powershell
python evaluation.py --all
```

默认数据目录是本文件夹下的 adult/，输出目录是本文件夹下的 outputs/，不依赖当前工作目录。
可以直接重复运行，已有输出目录会复用，同名结果文件会覆盖；不再需要指定新目录。
如需另存结果，仍可指定新的输出目录：

```powershell
python evaluation.py --all --work-dir outputs/run_02
```

也可以分开执行，但三个命令的 --work-dir 应相同：

```powershell
python preprocessing.py --work-dir outputs/run_03
python models.py --work-dir outputs/run_03
python evaluation.py --work-dir outputs/run_03
```

--data-dir 可以指定其他原始数据目录；显式相对路径也以项目目录为基准。
程序不捕获并隐藏异常；输入、模型或依赖缺失会报错并停止。

## 数据怎样传递

1. prepare_scheme1/2/3 执行原模型脚本的数据准备，输出包含 X_train、X_test、y_train、y_test 和未拟合 preprocess 的字典。
2. original_preprocessing 执行原 pre_clean.py 的三套变换和四张图，导出矩阵、标签和已拟合预处理器。
3. models.py 读取第一步的方案状态，在原有十折 CV 中拟合处理器和模型，保存测试标签、Pipeline、CV 结果、预测及概率。
4. evaluation.py 只读取模型输出，不需要原始数据或预处理文件，也不会重新拟合或预测。

processed_data.joblib 是预处理成员的展示结果，可按以下格式读取：

```python
processed = joblib.load(path)
processed["cleaned_data"]
processed["schemes"][1]["X_train"]
processed["schemes"][1]["y_train"]
```

模型不直接使用整份训练集上变换后的矩阵进行 CV，以保留原来的训练方式。
方案一原本保留 CV 分数数组，方案二、三原本只保留其均值，本版也保持如此。

## 保留的原问题

- 方案二、三将全部 X 转为 object，数值列列表为空，数值填补/标准化分支没有生效；未修复。
- 方案一删除缺失样本，测试人群与另两方案不同；未改变比较方式。
- 原比较脚本按测试 F1/AUC 排名；原排序保留。
- 官方训练/测试文件合并后重新分层划分；随机种子、比例、十折数及全部模型参数保持原样。
- 没有新增聚类、统计检验、重采样或特征工程。
- 原 API 获取结果改为本地同一份 Adult 原始数据；保留原 fetch 调用语句，仅在函数内部用新增绑定适配输入。
- 原 macOS 图片路径通过新增赋值绑定当前输出目录；原路径语句本身保留。
- 原预处理的 plt.show() 保留，默认 Agg 后端会提示非交互警告，但图仍保存。
- 运行时没有自动版本锁定或文件篡改检查；更换代码/数据/库版本后应重新运行完整流程，避免混用旧结果。如需保留旧结果，可指定新的输出目录。

## 验证

verification/ 保存本次原件保护、样本端到端、原始结果对比和真实数据预处理检查记录。
没有在项目中增加测试 Python 文件。采样测试不能作为完整 Adult 实验成绩。
真实全量模型训练较耗时，是否实际执行及验证范围以检查报告和交付说明为准。

本次检查已通过：

- 三个独立入口与 evaluation.py --all 在项目目录之外均成功运行，汇总指标一致。
- 240 行 Adult 样本上，三套预处理矩阵及标签与原脚本完全一致，12 组模型指标也与原脚本一致。
- 禁止 Pipeline.fit 后仍能完成评估，确认评估没有重新训练。
- 四张探索图、十二张模型及比较图均成功生成；缺输入/前置结果和已有输出均明确报错。
- 真实 48,842 行 Adult 已完成预处理；真实数据全量模型尚未重新训练。
- 原始文件 SHA-256 未变，原始注释均保留；项目仍只有三个 Python 文件。

详细记录：verification/integration_report.json、source_preservation.json 和 full_data_preprocessing_report.json。
默认 outputs/ 未生成，可直接按上面的命令开始正式流程；该精简版输出格式与旧版不同，请勿混用旧版中间文件。

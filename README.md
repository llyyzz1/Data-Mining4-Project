# Adult and Mushroom Data Mining Project

This CDS535 group project compares two UCI datasets through preprocessing,
classification, evaluation and Apriori association analysis. It includes both
original project implementations and a common, reproducible comparison using
Decision Tree, Random Forest, Logistic Regression and RBF SVM. The repository
contains the data, trained Adult artifacts, tables and figures needed to inspect
the saved results without repeating the expensive Adult training step.

All 10 project Python files were copied byte for byte. Folder names with spaces
are intentional: the comparison script uses these sibling paths by default.
No source algorithm, parameter, comment, variable or function was edited for
packaging. See [code preservation](CODE_PRESERVATION.json) and the checks below.

For current setup and execution instructions, use this root README. Original
subproject READMEs are retained as historical documentation; some describe an
older folder name, an already-fixed Adult numeric-type issue, or old validation
status. They do not override the current code or this root guide.

## Repository layout

```text
.
├── README.md
├── README_zh.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── CODE_PRESERVATION.json
├── FILE_MANIFEST.json
├── CHECKSUMS.sha256
├── verify_files.ps1
├── adult project/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── adult/                         UCI raw files and original descriptions
│   ├── outputs/preprocessing/         Three scheme states and descriptive data
│   ├── outputs/models/                Three trained states and Apriori outputs
│   ├── outputs/evaluation/            Saved metrics and figures
│   └── presentation/                  Existing Adult-only slides and script
├── Data mining project/
│   └── mushroom_preprocessing/
│       ├── preprocess_mushroom.py
│       ├── tree_models.py
│       ├── linear_svm_models.py
│       ├── evaluate_models.py
│       ├── model_tuning.py
│       ├── association_rules.py
│       ├── data/                      UCI raw files and original description
│       └── output/                    Prepared CSVs, metrics and figures
├── cross_dataset_comparison/
│   ├── compare_baselines.py
│   └── outputs/                       Matched-parameter 10-fold/holdout results
├── reports/                           Existing English and Chinese Word reports
└── docs/                              Data sources, limitations and upload guide
```

## Setup

Open a terminal at the repository root. The recorded execution environment is
Python 3.14.6 on Windows, with NumPy 2.4.6, pandas 3.0.3 and scikit-learn 1.9.0.
The root requirements file records the observed direct dependencies. Install
them in a separate virtual environment if needed:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then run:

```bash
python -m pip install -r requirements.txt
```

The Mushroom subfolder retains a different historical requirements file. Do
not install both sets into one environment. Use the root versions for this
package; exact byte-for-byte regeneration of historical plots is not promised.
Other Python/library versions have not been validated. Loading serialized
scikit-learn models should use the recorded environment. Installation from
package indexes is a separate step and was not performed during packaging.

## Adult

All commands below are issued from the repository root. Paths are quoted for
both PowerShell and ordinary shells. No path depends on a particular drive.

Run preprocessing, training and evaluation separately:

```bash
python -X utf8 -B "adult project/preprocessing.py"
python -X utf8 -B "adult project/models.py"
python -X utf8 -B "adult project/evaluation.py"
```

Run the entire Adult pipeline, including three schemes, 12 classifier runs with
training-set ten-fold CV, independent holdout evaluation and Apriori:

```bash
python -X utf8 -B "adult project/evaluation.py" --all
```

Training all Adult models is expensive. The package includes the saved models,
so `evaluation.py` can also run immediately without retraining. It reads the
stored predictions and writes metrics/figures; it does not fit classifiers.
Reruns reuse output folders and overwrite matching results. To keep a separate
Adult run, pass the same `--work-dir` to its three stages, or use:

```bash
python -X utf8 -B "adult project/evaluation.py" --all --work-dir outputs/new_run
```

That relative output path is resolved against `adult project/`. The comparison
script expects Adult results under its default `outputs/`, so its normal
comparison command will still read that baseline, not `outputs/new_run`.

## Mushroom

Preprocess the raw data:

```bash
python -X utf8 -B "Data mining project/mushroom_preprocessing/preprocess_mushroom.py"
```

Inspect the model definitions and data interfaces:

```bash
python -X utf8 -B "Data mining project/mushroom_preprocessing/tree_models.py"
python -X utf8 -B "Data mining project/mushroom_preprocessing/linear_svm_models.py"
```

These two commands inspect/print constructors; they do not perform a complete
training/evaluation experiment. Run the original experiment with:

```bash
python -X utf8 -B "Data mining project/mushroom_preprocessing/evaluate_models.py"
```

Unlike Adult's evaluation module, this Mushroom entry trains its classifiers
before evaluating them. Its existing `RUN_TUNING=True` and `RUN_APRIORI=True`
also run the historical five-fold grid search and Apriori. These settings have
not been changed. The two tasks can additionally run on their own:

```bash
python -X utf8 -B "Data mining project/mushroom_preprocessing/model_tuning.py"
python -X utf8 -B "Data mining project/mushroom_preprocessing/association_rules.py"
```

## Common cross-dataset baseline

```bash
python -X utf8 -B "cross_dataset_comparison/compare_baselines.py" --cv-jobs 2
```

This reuses Adult Scheme 2 and verifies the saved classifier parameters. It
trains Mushroom with the same four classifier configurations, training-set
stratified ten-fold CV and a separate 20% stratified holdout (seed 42). It does
not retrain Adult. Preprocessing remains dataset-specific.

All required inputs are already included. For a from-scratch reproduction,
first run the full Adult pipeline and Mushroom preprocessing, then this command.
Run the original Mushroom evaluation separately if reproducing its original
five-fold tuning experiment. It is not the common ten-fold baseline.

## Results to inspect

- `adult project/outputs/evaluation/adult_comparison/adult_all_schemes_results.csv`
  contains the 12 Adult classifier/scheme results.
- `Data mining project/mushroom_preprocessing/output/tables/model_metrics.csv`
  contains the original Mushroom holdout results.
- `cross_dataset_comparison/outputs/cross_dataset_metrics.csv` is the eight-row
  matched-configuration comparison used in the report.
- `cross_dataset_comparison/outputs/mushroom/cv_scores_10fold.csv` contains 40
  fold scores. `test_predictions.csv` and `confusion_matrices.csv` support review.
- The corresponding PNG figures are included alongside the tables.
- Apriori outputs are under Adult `outputs/models/adult_apriori/` and Mushroom
  `output/tables/apriori_*.csv`.

The saved comparison manifest says `status=complete`. Historical result CSVs,
JSON and older notes may contain absolute source paths from the original
machine; those strings describe provenance and are not required runtime paths.

## Data sources and scope

Both datasets were downloaded directly from UCI. Raw data and descriptions are
included. See [data provenance](docs/DATA_PROVENANCE.md) for DOI, attribution,
licensing and verified file hashes. See [known issues](docs/KNOWN_ISSUES.md)
for the distinction between historical tuning and the matched comparison,
missing-value interpretation, duplicate-record risk and report placeholders.
Apriori provides the unsupervised component; no new clustering is claimed.

## Team

| Member | Main responsibility |
| --- | --- |
| HUANG WENLONG | Adult ten-fold CV, independent evaluation protocol and joint cross-dataset analysis |
| LUO YIPENG | Adult preprocessing and data-quality descriptions |
| WEN YONGCHENG | Adult modeling and association analysis |
| LIU YONGZHI | Mushroom preprocessing and data-quality descriptions |
| FU SHIFANG | Mushroom modeling and association analysis |
| CHEN YISHU | Mushroom evaluation and joint cross-dataset analysis |

All six members participated in report writing. The included reports are exact
copies of the existing member versions. Their stored contribution estimates
and remaining placeholders have not been edited as part of packaging.

## Integrity and upload

On Windows, check the packaged files before running or editing anything:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify_files.ps1
```

On Linux with GNU coreutils, run `sha256sum -c CHECKSUMS.sha256` at the repository
root. Rerunning experiments can change output checksums legitimately; the
manifest records the packaged snapshot, not all future experiment runs.

The three Adult model files are about 34-36 MiB each. They are below GitHub's
100 MiB Git-push limit but exceed its 25 MiB browser-upload limit. Use Git push
for this complete directory, following [the Chinese upload guide](docs/GITHUB_UPLOAD_zh.md).
Git LFS is not required for the present files. Do not upload the outer delivery
ZIP as the repository contents: upload/commit the directory contents.

No repository was published or remote configured during packaging. No software
license for the team's own code has been invented; dataset licensing is
documented separately. Review course sharing requirements before publishing.

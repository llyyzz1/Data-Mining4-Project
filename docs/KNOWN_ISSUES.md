# Known issues and interpretation boundaries

1. The current Adult project is `adult project/`. Historical READMEs and some
   notes still mention `adult_project_three_files_simple/` and absolute local
   paths. Source code uses file-based paths and existing adapters. The old
   macOS figure-path assignment inside Adult preprocessing is followed by the
   existing output-path adaptation; it was not edited for this package.
2. Adult Scheme 2/3 restore numeric columns inside models.py using
   `infer_objects()`, rebuild the respective numeric/categorical pipelines and
   include Apriori. Old README statements that the issue remains unfixed or
   that no unsupervised component exists are historical and no longer accurate.
   Existing output directories are reusable; the old directory-protection text
   in the original README introduction is also outdated.
3. Adult and Mushroom original requirements files record different versions.
   Use the root requirements for the observed shared environment. This is a
   direct-dependency snapshot, not proof of a fresh pip install or a complete
   transitive lock. Serialized models require compatible library versions.
4. The Mushroom historical experiment uses a calibrated linear SVM and a
   five-fold tuning grid. The common comparison uses RBF SVM, fixed Adult
   classifier configurations, training-only ten-fold CV and a 20% independent
   holdout. Do not combine their results as though they were the same experiment.
5. Mushroom audit reads `?` as missing, but its modeling CSV/reader retains it
   as a string category. The downstream SimpleImputer does not replace these
   literal question marks with modes. The current comparison preserves this.
6. Adult Scheme 1 removes incomplete records and has a different test
   population from Schemes 2/3. Adult also retains duplicate records, so possible
   overlap between train/test feature patterns remains a stated limitation.
   Adult Scheme 2/3 save CV means rather than all fold scores; do not infer a
   saved ten-fold standard deviation from these means.
7. Apriori is exploratory full-data analysis. Its confidence is not independent
   test accuracy. Dataset thresholds differ, so rule counts alone do not rank
   the two datasets or algorithms. No additional clustering is included.
8. SVC(probability=True) emits a deprecation warning in the recorded sklearn
   version. Original warnings/settings have not been suppressed or altered.
   Adult preprocessing retains plt.show() under an Agg backend; noninteractive
   warnings can appear while the figures are still saved.
9. Rerunning commands overwrites matching outputs in the COPY. The packaged
   hashes describe the original delivered snapshot, and legitimately change
   after results are regenerated. Original source directories outside this
   upload directory are not used at runtime and were not modified.
10. Both Word reports are existing copies, not a new final submission. Student
    IDs/team details may still need filling, stored equal contribution estimates
    have not been replaced by the later suggested allocation, and the template
    reports exceeded the course page target in the last review. No outdated
    PDF was relabeled as a current report. Export the finalized Word to PDF
    separately. Existing presentation files cover Adult evaluation only.
11. Python experiments and reproducibility are documented. No Weka execution
    is invented. The course Weka demonstration and full-team oral presentation
    remain separate preparation tasks unless the instructor approves otherwise.
12. Historical README requirements asserting byte-for-byte reproduction of all
    outputs are not validated by this packaging exercise. Timing, paths, fonts,
    image metadata and library versions can change artifacts on reruns.

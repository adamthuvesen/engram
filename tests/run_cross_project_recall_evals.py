"""Compatibility entrypoint for ``engram.evaluation.cross_project``."""

from engram.evaluation.cross_project import (
    BASELINE_QUALITY_SCORE as BASELINE_QUALITY_SCORE,
    DATASET_PATH as DATASET_PATH,
    LabeledQuery as LabeledQuery,
    MIN_QUALITY_SCORE as MIN_QUALITY_SCORE,
    MIN_RELATIVE_GAIN as MIN_RELATIVE_GAIN,
    QueryResult as QueryResult,
    Summary as Summary,
    evaluate as evaluate,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())

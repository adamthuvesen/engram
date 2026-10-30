"""Compatibility entrypoint for ``engram.evaluation.memory_audit``."""

from engram.evaluation.memory_audit import (
    DATASET_PATH as DATASET_PATH,
    EvalSummary as EvalSummary,
    ExpectedIssue as ExpectedIssue,
    MAX_BURDEN_MULTIPLIER as MAX_BURDEN_MULTIPLIER,
    MIN_NEW_PRECISION as MIN_NEW_PRECISION,
    MIN_NEW_RECALL as MIN_NEW_RECALL,
    MIN_RECALL_GAIN as MIN_RECALL_GAIN,
    MetricBlock as MetricBlock,
    evaluate as evaluate,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())

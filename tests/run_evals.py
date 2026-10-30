"""Compatibility entrypoint for ``engram.evaluation.recall``."""

from engram.evaluation.recall import (
    DATASET_PATH as DATASET_PATH,
    LabeledQuery as LabeledQuery,
    MIN_HIT_RATE as MIN_HIT_RATE,
    MIN_LEXICAL_FRACTION as MIN_LEXICAL_FRACTION,
    MIN_MRR as MIN_MRR,
    MIN_RECALL_AT_1 as MIN_RECALL_AT_1,
    MIN_RECALL_AT_5 as MIN_RECALL_AT_5,
    QueryResult as QueryResult,
    Summary as Summary,
    evaluate as evaluate,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())

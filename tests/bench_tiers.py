"""Compatibility entrypoint for ``engram.evaluation.tiers``."""

from engram.evaluation.tiers import (
    FACTS as FACTS,
    MONTH_AGO as MONTH_AGO,
    NOW as NOW,
    QUERIES as QUERIES,
    WEEK_AGO as WEEK_AGO,
    main as main,
)

if __name__ == "__main__":
    raise SystemExit(main())

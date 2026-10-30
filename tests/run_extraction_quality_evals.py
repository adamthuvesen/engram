"""Compatibility entrypoint for ``engram.evaluation.extraction_quality``."""

from engram.evaluation.extraction_quality import (
    CanonicalMemoryUnit as CanonicalMemoryUnit,
    CaseScore as CaseScore,
    DATASET_PATH as DATASET_PATH,
    EvalSummary as EvalSummary,
    ExtractedCard as ExtractedCard,
    ExtractionEvalCase as ExtractionEvalCase,
    ExtractionEvalDataset as ExtractionEvalDataset,
    MAX_FRAGMENTATION as MAX_FRAGMENTATION,
    MIN_DURABLE_CLAIM_COVERAGE as MIN_DURABLE_CLAIM_COVERAGE,
    MIN_EMITTED_CARD_PRECISION as MIN_EMITTED_CARD_PRECISION,
    POLICY_REQUIRED_CLAUSES as POLICY_REQUIRED_CLAUSES,
    ProviderOutputs as ProviderOutputs,
    evaluate as evaluate,
    extract_live_provider_outputs as extract_live_provider_outputs,
    load_dataset as load_dataset,
    main as main,
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run the configured provider. Requires ENGRAM_EVAL_PROVIDER=1.",
    )
    raise SystemExit(main(live=parser.parse_args().live))

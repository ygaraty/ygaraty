# Excluded timing trials

These first two trials completed training, but the observer stopped at the
coordinator version increment before the last evaluation/checkpoint completed.
Their timing curves omit the final evaluation and are excluded from the final
comparison. The complete final model evaluations remain in `metrics`.
The two arms are rerun after fixing transaction-boundary observation.

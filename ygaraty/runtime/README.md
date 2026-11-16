# Runtime

Local forward, backward and optimizer execution, plus task scheduling, capture
and memory-budget helpers. `evaluate` evaluates a Tape; `execute_plan` runs its
compiled task plan. This implementation depends on public core modules and
NumPy, with no application host requirement.

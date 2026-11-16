"""Execute a topologically ordered plan locally, releasing dead intermediates.

No scheduler, network or tensor store is needed when a participant holds the
whole model. Task/Input and the numerical kernels remain the shared contracts.
"""

from collections import Counter

from ygaraty.runtime.kernels import execute


def execute_plan(tasks, inputs, keep):
    """Return only `keep`; inputs are borrowed and kernels must not mutate
    them.

    Unlike the parity executor, this does not retain the entire forward and
    backward graph. Plans must be ordered and contain ordinary (ungrouped)
    tasks.
    """
    store = dict(inputs)
    keep = set(keep)
    remaining = Counter(dep for task in tasks for dep in task.dependencies)
    known = set(store)
    for task in tasks:
        if task.id in known or not task.dependencies <= known:
            raise ValueError(f"Invalid or unordered local plan at {task.id}")
        known.add(task.id)
    if not keep <= known:
        raise ValueError("Local plan is missing requested outputs")
    for key in list(store):
        if not remaining[key] and key not in keep:
            del store[key]
    for task in tasks:
        value = execute(task.op, task.arrays(store), task.attrs)
        if value.shape != tuple(task.shape):
            raise ValueError(f"Invalid local output shape for {task.id}")
        store[task.id] = value
        for dep in task.dependencies:
            remaining[dep] -= 1
            if not remaining[dep] and dep not in keep:
                del store[dep]
        if not remaining[task.id] and task.id not in keep:
            del store[task.id]
    return {key: store[key] for key in keep}

"""Export a saved Ygaraty task plan as one readable computation per line."""

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TaskDescription = Mapping[str, Any]


def _format_input(reference: Mapping[str, Any]) -> str:
    value = json.dumps(reference["value"])
    if reference["region"] is not None:
        row_start, row_stop, column_start, column_stop = reference["region"]
        value += f"[{row_start}:{row_stop}, {column_start}:{column_stop}]"
    return value


def _format_task(number: int, task: TaskDescription) -> str:
    inputs = ", ".join(_format_input(ref) for ref in task["inputs"])
    shape = " x ".join(map(str, task["shape"]))
    attrs = json.dumps(task["attrs"], sort_keys=True, separators=(",", ":"))
    return (
        f"{number:05d} | {json.dumps(task['id'])} = {task['op']}({inputs}) "
        f"-> [{shape}] | attrs={attrs}\n"
    )


class PlanExport:
    """Validated, detached text/JSONL renderings of an ordered task list.

    This validates task identity and dependency order, not numerical semantics
    or admission budgets. Construction finishes before any output is opened.
    """

    def __init__(self, tasks: Sequence[TaskDescription]):
        ids = [task["id"] for task in tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate task IDs")
        all_ids, seen, external = set(ids), set(), set()
        text_lines, json_lines = [], []
        for number, task in enumerate(tasks, 1):
            for reference in task["inputs"]:
                ident = reference["value"]
                if ident in all_ids and ident not in seen:
                    raise ValueError("Plan is not in dependency order")
                if ident not in all_ids:
                    external.add(ident)
            text_lines.append(_format_task(number, task))
            json_lines.append(json.dumps(task, separators=(",", ":")) + "\n")
            seen.add(task["id"])
        self._text = tuple(text_lines)
        self._jsonl = tuple(json_lines)
        self._external_inputs = tuple(sorted(external))

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "tasks": len(self._text),
            "external_inputs": list(self._external_inputs),
        }

    def write(self, text_path: str | Path, jsonl_path: str | Path) -> None:
        """Create outputs exclusively; existing files are never
        overwritten.

        The two creations are sequential, not a transaction. A failure
        creating the second file can leave the first output in place.
        """
        with Path(text_path).open("x") as stream:
            stream.writelines(self._text)
        with Path(jsonl_path).open("x") as stream:
            stream.writelines(self._jsonl)


def export_plan(
    source: str | Path, text_path: str | Path, jsonl_path: str | Path
) -> dict[str, Any]:
    """Compatibility entry point for exporting a saved JSON task list."""
    plan = PlanExport(json.loads(Path(source).read_text()))
    plan.write(text_path, jsonl_path)
    return plan.summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_plan(args.plan, args.text, args.jsonl), indent=2))


if __name__ == "__main__":
    main()

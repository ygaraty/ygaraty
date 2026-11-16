"""Small strict validators for immutable in-process contract values."""

from enum import Enum


def text(value: str, name: str, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(
            f"{name} must be nonempty text, at most {maximum} chars"
        )


def integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def member(value: Enum, kind: type[Enum]) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"Expected {kind.__name__}")


def immutable_items(value: tuple, kind: type, name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, kind) for item in value
    ):
        raise ValueError(f"{name} must be a tuple of {kind.__name__}")

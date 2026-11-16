"""Safe local artifact lookup shared by inference backends."""

from ygaraty.inference_support.artifacts import local_file

from pathlib import Path




def bundle_capacity(data):
    if not isinstance(data, dict) or data.get("format") != "ygaraty-webgpu-v1":
        raise ValueError("Unsupported bundle format")
    options = data.get("options")
    if not isinstance(options, dict) or not isinstance(
        data.get("stages"), list
    ):
        raise ValueError("Invalid bundle metadata")
    capacity = options.get("capacity")
    if type(capacity) is not int or not 1 <= capacity <= 2**31 - 1:
        raise ValueError("Invalid bundle context capacity")
    for stage in data.get("stages", []):
        if not isinstance(stage, dict) or not isinstance(
            stage.get("fixture"), dict
        ):
            raise ValueError("Invalid prepared stage")
        limit = stage["fixture"].get("max_positions", capacity)
        if type(limit) is not int or limit < capacity:
            raise ValueError(
                "Bundle context exceeds a prepared stage capacity"
            )
    return capacity

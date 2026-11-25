"""One CLI for the public compiler, training library and WebGPU server."""

import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "train":
        from ygaraty.training.__main__ import main as train

        return train(args[1:])
    if args and args[0] == "compile":
        from ygaraty.compiler.export import main as compile_plan

        return compile_plan(args[1:])
    from ygaraty.backends.webgpu.cli import main as server

    if args in (["--help"], ["-h"]):
        try:
            server(args)
        except SystemExit as error:
            if error.code:
                raise
        print("\nCore commands:")
        print("  compile    Export a saved Tape plan (compile --help)")
        print("  train      DiLoCo training (train serve --help for browser contributors)")
        return 0
    return server(args)

# Copyright (c) 2026 Graphics Group @ Università della Svizzera italiana (USI)
# https://www.pdf.inf.usi.ch/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""gfields.cli — the unified command-line dispatcher.

Usage:
    gfields <tool> [subcommand] [options]
    python -m gfields <tool> [subcommand] [options]

Run ``gfields`` with no arguments for the tool list, ``gfields <tool>
--help`` for its subcommands, and ``gfields <tool> <subcommand> --help``
for the full argument reference. Each tool lives in its home package
(``gfields.viewer``, ``gfields.voxelizer``, ``gfields.lod``,
``gfields.training``); this module only routes to them.

Training and rendering are **not** here — they are the two promoted
top-level scripts ``python -m gfields.train`` and ``python -m gfields.render``
(the latter also handles ``render video``). This dispatcher only covers the
auxiliary tools.
"""

from __future__ import annotations

import runpy
import sys


def _tool(description, subcommands, default=None):
    return {"description": description, "subcommands": subcommands,
            "default": default}


TOOLS = {
    "viewer": _tool(
        "Interactive polyscope viewer (progressive accumulation, LOD slider)",
        {"": ("gfields.viewer.cli", "")}, default=""),
    "voxelize": _tool(
        "Voxelize primitive mixtures into dense grids", {
            "asset": ("gfields.voxelizer.voxelize", "Voxelize an asset into a dense grid (default)"),
            "compare": ("gfields.voxelizer.compare", "Voxelize + render grid vs the primitive mixture"),
            "grid": ("gfields.references.grid", "Render a dense .vol/.npy voxel grid"),
        }, default="asset"),
    "decompose": _tool(
        "Steerable decomposition, LOD partitioning, clamping, pruning", {
            "steerable": ("gfields.lod.decompose", "Frequency x orientation steerable decomposition"),
            "lod-partition": ("gfields.lod.decompose", "Partition primitives into frequency-band LOD levels"),
            "prune": ("gfields.training.post_pruning", "PSNR-budget primitive pruning"),
            "clamp": ("gfields.training.adaptive", "Adaptive per-primitive extent clamping"),
        }),
}


def _print_overview() -> None:
    print(__doc__)
    print("tools:")
    for name, tool in TOOLS.items():
        print(f"  {name:<12} {tool['description']}")


def _print_tool_help(name: str) -> None:
    tool = TOOLS[name]
    print(f"gfields {name} — {tool['description']}\n")
    print("subcommands:")
    for sub, (_, desc) in tool["subcommands"].items():
        label = sub or "(none — options go directly to the tool)"
        print(f"  {label:<28} {desc}".rstrip())
    if tool["default"]:
        print(f"\ndefault subcommand: {tool['default']}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _print_overview()
        return
    name, argv = argv[0], argv[1:]
    if name not in TOOLS:
        raise SystemExit(f"unknown tool {name!r} (available: {', '.join(TOOLS)})")
    tool = TOOLS[name]

    single = list(tool["subcommands"]) == [""]
    if single:
        sub, rest = "", argv
    elif argv and argv[0] in tool["subcommands"]:
        sub, rest = argv[0], argv[1:]
    elif argv and argv[0] in ("-h", "--help"):
        _print_tool_help(name)
        return
    elif tool["default"] is not None:
        sub, rest = tool["default"], argv
    else:
        _print_tool_help(name)
        raise SystemExit(1 if not argv else f"\nunknown subcommand {argv[0]!r}")

    module = tool["subcommands"][sub][0]
    label = f"gfields {name} {sub}".rstrip()
    sys.argv = [label, *rest]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()

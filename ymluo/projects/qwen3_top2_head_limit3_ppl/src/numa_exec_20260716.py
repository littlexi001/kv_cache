from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path


class Bitmask(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_ulong),
        ("maskp", ctypes.POINTER(ctypes.c_ulong)),
    ]


def parse_cpu_list(spec: str) -> set[int]:
    cpus: set[int] = set()
    for item in spec.strip().split(","):
        if "-" in item:
            start, end = map(int, item.split("-", maxsplit=1))
            cpus.update(range(start, end + 1))
        elif item:
            cpus.add(int(item))
    return cpus


def node_cpu_list(node: int) -> set[int]:
    path = Path(f"/sys/devices/system/node/node{node}/cpulist")
    if not path.is_file():
        raise ValueError(f"NUMA node {node} is unavailable")
    return parse_cpu_list(path.read_text(encoding="ascii"))


def bind_numa(node: int) -> dict[str, object]:
    library = ctypes.CDLL("libnuma.so.1", use_errno=True)
    library.numa_available.restype = ctypes.c_int
    library.numa_run_on_node.argtypes = [ctypes.c_int]
    library.numa_run_on_node.restype = ctypes.c_int
    library.numa_allocate_nodemask.restype = ctypes.POINTER(Bitmask)
    library.numa_bitmask_clearall.argtypes = [ctypes.POINTER(Bitmask)]
    library.numa_bitmask_setbit.argtypes = [ctypes.POINTER(Bitmask), ctypes.c_uint]
    library.numa_bitmask_setbit.restype = ctypes.POINTER(Bitmask)
    library.numa_set_membind.argtypes = [ctypes.POINTER(Bitmask)]
    library.numa_set_membind.restype = None
    library.numa_get_membind.restype = ctypes.POINTER(Bitmask)

    if library.numa_available() < 0:
        raise RuntimeError("libnuma reports that NUMA is unavailable")
    cpus = node_cpu_list(node)
    if library.numa_run_on_node(node) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    mask = library.numa_allocate_nodemask()
    if not mask:
        raise MemoryError("numa_allocate_nodemask failed")
    library.numa_bitmask_clearall(mask)
    library.numa_bitmask_setbit(mask, node)
    library.numa_set_membind(mask)

    effective = library.numa_get_membind()
    bits_per_word = ctypes.sizeof(ctypes.c_ulong) * 8
    word = node // bits_per_word
    bit = node % bits_per_word
    bound = bool(effective.contents.maskp[word] & (1 << bit))
    if not bound:
        raise RuntimeError(f"memory policy does not include requested NUMA node {node}")
    affinity = set(os.sched_getaffinity(0))
    if not affinity or not affinity.issubset(cpus):
        raise RuntimeError(
            f"CPU affinity {sorted(affinity)} is not restricted to NUMA node {node}"
        )
    return {
        "numa_node": node,
        "cpu_affinity": sorted(affinity),
        "memory_policy_contains_node": bound,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("a command is required after --")
    binding = bind_numa(args.node)
    print(json.dumps(binding, sort_keys=True), flush=True)
    os.environ["NUMA_BIND_NODE"] = str(args.node)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()

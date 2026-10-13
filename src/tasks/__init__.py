"""ZLAB task registry: Z01_std_add_head tracking + velocity tasks."""

import importlib

_TASK_GROUPS = {
    "tracking": (
        "src.tasks.tracking.config.z01_std_add_head",
    ),
    "velocity": (
        "src.tasks.velocity.config.z01_std_add_head",
    ),
}

for _group_modules in _TASK_GROUPS.values():
    for module_name in _group_modules:
        importlib.import_module(module_name)

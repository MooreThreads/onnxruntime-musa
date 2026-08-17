"""ONNX Runtime MUSA Plugin Execution Provider Python package."""

from __future__ import annotations

import json
import pathlib
from typing import Any

__all__ = [
    "get_ep_name",
    "get_ep_names",
    "get_library_path",
    "get_manifest",
    "get_manifest_path",
    "make_provider_options",
]

_module_dir = pathlib.Path(__file__).parent


def get_library_path() -> str:
    """Return the path to the MUSA plugin EP shared library."""
    candidate_paths = [
        _module_dir / "onnxruntime_providers_musa_plugin.dll",
        _module_dir / "libonnxruntime_providers_musa_plugin.so",
    ]
    paths = [p for p in candidate_paths if p.is_file()]
    if len(paths) != 1:
        raise RuntimeError(
            f"Expected exactly one MUSA plugin EP library in {_module_dir}, "
            f"found {len(paths)}: {[p.name for p in paths]}"
        )
    return str(paths[0])


def get_manifest_path() -> str:
    """Return the path to the installed MUSA Plugin EP manifest."""
    path = _module_dir / "manifest.json"
    if not path.is_file():
        raise RuntimeError(f"MUSA Plugin EP manifest not found at {path}")
    return str(path)


def get_manifest() -> dict[str, Any]:
    """Load and return the installed MUSA Plugin EP compatibility manifest."""
    with pathlib.Path(get_manifest_path()).open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict):
        raise RuntimeError("MUSA Plugin EP manifest must contain a JSON object")
    return manifest


def get_ep_name() -> str:
    """Return the MUSA plugin Execution Provider name."""
    return "MUSAExecutionProvider"


def get_ep_names() -> list[str]:
    """Return a list of EP names provided by this plugin."""
    return [get_ep_name()]


def _as_bool_option(value: bool) -> str:
    if not isinstance(value, bool):
        raise TypeError("MUSA provider boolean options must be bool values")
    return "1" if value else "0"


def _as_stream_address(value: int | Any) -> int:
    if isinstance(value, int):
        address = value
    elif hasattr(value, "value"):
        address = int(value.value or 0)
    else:
        address = int(value)
    if address <= 0:
        raise ValueError("user_compute_stream must be a non-null musaStream_t address")
    return address


def make_provider_options(
    *,
    device_id: int = 0,
    user_compute_stream: int | Any | None = None,
    use_ep_level_unified_stream: bool | None = None,
    do_copy_in_default_stream: bool = True,
    precision_policy: str = "strict",
    dtype_diagnostics: bool = False,
) -> dict[str, str]:
    """Build MUSAExecutionProvider options for add_provider_for_devices().

    ORT Plugin EP V2 provider options are key/value strings, so this helper
    converts the stable MUSA option surface into the representation expected by
    SessionOptions.add_provider_for_devices().
    """
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    if precision_policy not in {"strict", "report"}:
        raise ValueError("precision_policy must be 'strict' or 'report'")

    options = {
        "device_id": str(device_id),
        "do_copy_in_default_stream": _as_bool_option(do_copy_in_default_stream),
        "precision_policy": precision_policy,
        "dtype_diagnostics": _as_bool_option(dtype_diagnostics),
    }

    if user_compute_stream is not None:
        address = _as_stream_address(user_compute_stream)
        options["has_user_compute_stream"] = "1"
        options["user_compute_stream"] = str(address)
        options["use_ep_level_unified_stream"] = "1"
    elif use_ep_level_unified_stream is not None:
        options["use_ep_level_unified_stream"] = _as_bool_option(
            use_ep_level_unified_stream
        )

    return options

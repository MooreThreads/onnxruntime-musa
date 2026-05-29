from __future__ import annotations

import json

import onnxruntime as ort
import onnxruntime_musa as musa_ep


def main() -> None:
    ort.register_execution_provider_library(musa_ep.get_ep_name(), musa_ep.get_library_path())
    devices = ort.get_ep_devices()
    print(json.dumps(
        [
            {
                "ep_name": getattr(device, "ep_name", None),
                "device": str(getattr(device, "device", "")),
                "metadata": getattr(device, "ep_metadata", None),
                "options": getattr(device, "ep_options", None),
            }
            for device in devices
        ],
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import onnxruntime as ort
import onnxruntime_musa as musa_ep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    ort.register_execution_provider_library(musa_ep.get_ep_name(), musa_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == musa_ep.get_ep_name()]
    if not devices:
        raise RuntimeError(f"{musa_ep.get_ep_name()} is not visible")

    sess_options = ort.SessionOptions()
    sess_options.add_provider_for_devices(devices[:1], {})
    session = ort.InferenceSession(args.model, sess_options=sess_options)
    print(session.get_providers())


if __name__ == "__main__":
    main()

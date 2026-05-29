#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORT_ROOT = Path("/home/workspace/onnxruntime")
DEFAULT_ORT_BUILD_ROOT = DEFAULT_ORT_ROOT / "build_musa_wheel_py311/Release"


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd or REPO_ROOT), env=env)


def build_plugin(args: argparse.Namespace) -> Path:
    build_dir = REPO_ROOT / "build" / args.config
    build_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            "cmake",
            "-S",
            str(REPO_ROOT),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.config}",
            f"-DORT_ROOT={args.ort_root}",
            f"-DORT_BUILD_ROOT={args.ort_build_root}",
            f"-DMUSA_HOME={args.musa_home}",
        ]
    )
    run(["cmake", "--build", str(build_dir), "--config", args.config, "-j", str(args.jobs)])
    return build_dir


def build_wheel(args: argparse.Namespace, binary_dir: Path) -> Path:
    dist_dir = REPO_ROOT / "dist"
    run(
        [
            sys.executable,
            str(REPO_ROOT / "musa/ep/python/build_wheel.py"),
            "--binary_dir",
            str(binary_dir),
            "--version",
            (REPO_ROOT / "VERSION_NUMBER").read_text(encoding="utf-8").strip(),
            "--package_name",
            args.package_name,
            "--output_dir",
            str(dist_dir),
        ]
    )
    wheels = sorted(dist_dir.glob(f"{args.package_name.replace('-', '_')}-*.whl"))
    if not wheels:
        raise RuntimeError(f"No wheel found in {dist_dir}")
    return wheels[-1]


def install_wheel(wheel: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheel)])


def smoke(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["MUSA_VISIBLE_DEVICES"] = str(args.device)
    ld_parts = [
        str(REPO_ROOT / "build" / args.config),
        "/usr/local/musa/lib",
        "/usr/local/musa/lib64",
        env.get("LD_LIBRARY_PATH", ""),
    ]
    env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_parts if p])
    run([sys.executable, str(REPO_ROOT / "musa/samples/list_ep_devices.py")], env=env)
    run([sys.executable, str(REPO_ROOT / "musa/model_test/smoke_matmul.py")], env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and test onnxruntime-musa")
    parser.add_argument("--config", default="Release")
    parser.add_argument("--ort-root", type=Path, default=DEFAULT_ORT_ROOT)
    parser.add_argument("--ort-build-root", type=Path, default=DEFAULT_ORT_BUILD_ROOT)
    parser.add_argument("--musa-home", type=Path, default=Path("/usr/local/musa"))
    parser.add_argument("--package-name", default="onnxruntime-musa")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--device", default="5")
    parser.add_argument("--build-plugin", action="store_true")
    parser.add_argument("--build-wheel", action="store_true")
    parser.add_argument("--install-wheel", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean:
        shutil.rmtree(REPO_ROOT / "build", ignore_errors=True)
        shutil.rmtree(REPO_ROOT / "dist", ignore_errors=True)

    binary_dir = REPO_ROOT / "build" / args.config / "Release"
    wheel = None

    if args.build_plugin or args.build_wheel or args.install_wheel or args.smoke:
        binary_dir = build_plugin(args)
    if args.build_wheel or args.install_wheel:
        wheel = build_wheel(args, binary_dir)
    if args.install_wheel:
        assert wheel is not None
        install_wheel(wheel)
    if args.smoke:
        smoke(args)


if __name__ == "__main__":
    main()

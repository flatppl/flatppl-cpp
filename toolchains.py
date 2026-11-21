"""Resolve local tools or bootstrap pinned tools when asked."""

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

FLATPPL_REVISION = "5d4cb2050fb9bbf3c7f715eab95e8201400c05cc"
IREE_VERSION = "3.11.0"
IREE_REVISION = "e4a3b0405d7d23554da26403658d0e8c3c5ecf25"
IREE_ASSET_PREFIX = "iree_base_compiler-3.11.0rc20260316-cp312-abi3-"
IREE_ASSETS = {
    ("Darwin", "universal2"): (
        "macosx_13_0_universal2",
        "7647c31f63b3dad250005ab90d724fe682146c664f8a39cbe3996011253eebd3",
    ),
    ("Linux", "aarch64"): (
        "manylinux_2_27_aarch64.manylinux_2_28_aarch64",
        "c1eca18c4cbe0701c81497e1e4b709aa470dd14a73fc138839f88afd9b9dc68d",
    ),
    ("Linux", "x86_64"): (
        "manylinux_2_27_x86_64.manylinux_2_28_x86_64",
        "0cf85af2a59ce9d0aca706c50a827e7c53c0ed5dcfa853f8c6f7b957142dd6b5",
    ),
    ("Windows", "x86_64"): (
        "win_amd64",
        "b80efa140f342ea57e93b38e1494408f6646a059c5e2bd42a81740c45d8075f4",
    ),
}
EXE = ".exe" if os.name == "nt" else ""


class BuildError(Exception):
    pass


def run(arguments, *, quiet=False, env=None):
    process = subprocess.run(
        arguments,
        check=False,
        text=True,
        capture_output=quiet,
        env=env,
    )
    if process.returncode:
        if quiet:
            sys.stderr.write(process.stdout + process.stderr)
        raise BuildError(f"{Path(arguments[0]).name} exited with {process.returncode}")


@contextmanager
def lock_directory(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise BuildError(f"another build holds {path}") from error
    try:
        yield
    finally:
        path.rmdir()


def checkout(destination, url, revision):
    if destination.is_dir():
        return
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "source"
        run(["git", "init", stage], quiet=True)
        run(["git", "-C", stage, "fetch", "--depth=1", url, revision])
        run(["git", "-C", stage, "checkout", "--detach", "FETCH_HEAD"], quiet=True)
        stage.rename(destination)


def flatppl_compiler(path, fetch, cache):
    if path:
        executable = path.resolve()
    elif fetch:
        source = cache / f"flatppl-{FLATPPL_REVISION}"
        target = cache / f"cargo-{FLATPPL_REVISION}"
        executable = target / "release" / ("flatppl" + EXE)
        with lock_directory(cache / "flatppl.lock"):
            if not executable.is_file():
                checkout(
                    source,
                    "https://github.com/flatppl/flatppl-rust.git",
                    FLATPPL_REVISION,
                )
                run(
                    [
                        "cargo",
                        "build",
                        "--manifest-path",
                        source / "Cargo.toml",
                        "--locked",
                        "--release",
                        "-p",
                        "flatppl-cli",
                        "--bin",
                        "flatppl",
                    ],
                    env={**os.environ, "CARGO_TARGET_DIR": str(target)},
                )
    else:
        raise BuildError("choose --flatppl or --fetch-flatppl")
    if not executable.is_file():
        raise BuildError(f"FlatPPL executable not found: {executable}")
    return executable


def iree_asset():
    system = platform.system()
    machine = platform.machine().lower()
    if machine == "amd64":
        machine = "x86_64"
    if system == "Darwin" and machine in ("arm64", "x86_64"):
        machine = "universal2"
    key = (system, machine)
    if key not in IREE_ASSETS:
        raise BuildError(
            f"no pinned IREE compiler for {system} {machine}; select a local compiler"
        )
    suffix, digest = IREE_ASSETS[key]
    return IREE_ASSET_PREFIX + suffix + ".whl", digest


def extract_compiler(archive, destination, expected_sha256):
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected_sha256:
        raise BuildError(f"IREE compiler checksum mismatch: {archive.name}")
    with ZipFile(archive) as wheel:
        wheel.extractall(destination)
        for item in wheel.infolist():
            if item.external_attr >> 16 & stat.S_IXUSR:
                path = destination / item.filename
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
    compiler = destination / "iree" / "compiler" / "_mlir_libs" / ("iree-compile" + EXE)
    if not compiler.is_file():
        raise BuildError("IREE release archive lacks iree-compile")
    return compiler


def download_compiler(cache):
    asset, digest = iree_asset()
    installed = cache / ("compiler-" + digest[:16])
    executable = installed / "iree" / "compiler" / "_mlir_libs" / ("iree-compile" + EXE)
    if not executable.is_file():
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            stage = Path(temp)
            archive = stage / asset
            url = f"https://github.com/iree-org/iree/releases/download/v{IREE_VERSION}/{asset}"
            with urlopen(url, timeout=60) as response, archive.open("wb") as out:
                shutil.copyfileobj(response, out)
            extract_compiler(archive, stage / "compiler", digest)
            (stage / "compiler").rename(installed)
    return executable


def compiler_build(source, cache, jobs):
    key = hashlib.sha256(os.fsencode(source)).hexdigest()[:16]
    build = cache / f"iree-compiler-{key}"
    options = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        "-DIREE_BUILD_TESTS=OFF",
        "-DIREE_BUILD_SAMPLES=OFF",
        "-DIREE_BUILD_BINDINGS_TFLITE=OFF",
        "-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF",
        "-DIREE_INPUT_TORCH=OFF",
        "-DIREE_ENABLE_LIBBACKTRACE=OFF",
        "-DIREE_INPUT_STABLEHLO=ON",
        "-DIREE_TARGET_BACKEND_DEFAULTS=OFF",
        "-DIREE_TARGET_BACKEND_LLVM_CPU=ON",
        "-DIREE_HAL_DRIVER_DEFAULTS=OFF",
    ]
    run(["cmake", "-S", source, "-B", build, *options])
    parallel = ["--parallel", str(jobs)] if jobs else []
    run(
        [
            "cmake",
            "--build",
            build,
            "--config",
            "Release",
            *parallel,
            "--target",
            "iree-compile",
            "iree-lld",
        ]
    )
    for directory in (build / "tools", build / "tools" / "Release"):
        executable = directory / ("iree-compile" + EXE)
        if executable.is_file():
            return executable
    raise BuildError("could not locate iree-compile in the IREE build")


def iree_tools(args, cache):
    if args.iree_dir:
        source = args.iree_dir.resolve()
    elif args.fetch_iree:
        source = cache / f"iree-{IREE_REVISION}"
    else:
        raise BuildError("choose --iree-dir or --fetch-iree")

    with lock_directory(cache / "iree.lock"):
        if args.fetch_iree:
            checkout(source, "https://github.com/iree-org/iree.git", IREE_REVISION)
            # IREE configures benchmark whenever runtime threading is enabled.
            submodules = [
                "third_party/benchmark",
                "third_party/flatcc",
                "third_party/printf",
            ]
            if args.build_iree_from_source:
                submodules += [
                    "third_party/llvm-project",
                    "third_party/stablehlo",
                    "third_party/musl",
                ]
            run(
                [
                    "git",
                    "-C",
                    source,
                    "submodule",
                    "update",
                    "--init",
                    "--depth=1",
                    *submodules,
                ]
            )
        if not (source / "runtime" / "src" / "iree" / "runtime" / "api.h").is_file():
            raise BuildError(f"IREE source not found: {source}")
        if args.build_iree_from_source:
            compiler = compiler_build(source, cache, args.jobs)
        elif args.iree_compile:
            compiler = args.iree_compile.resolve()
        elif args.fetch_iree:
            compiler = download_compiler(cache)
        else:
            executable = shutil.which("iree-compile")
            if not executable:
                raise BuildError("set --iree-compile or put iree-compile on PATH")
            compiler = Path(executable)
    if not compiler.is_file():
        raise BuildError(f"IREE executable not found: {compiler}")
    return source, compiler

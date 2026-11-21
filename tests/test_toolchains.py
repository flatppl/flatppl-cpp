import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile, ZipInfo

import toolchains


class ToolchainTests(unittest.TestCase):
    def test_local_tools_do_not_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            flatppl = root / "flatppl"
            flatppl.touch()
            compiler = root / "iree-compile"
            compiler.touch()
            api = root / "iree" / "runtime" / "src" / "iree" / "runtime" / "api.h"
            api.parent.mkdir(parents=True)
            api.touch()
            args = SimpleNamespace(
                iree_dir=root / "iree",
                fetch_iree=False,
                build_iree_from_source=False,
                iree_compile=compiler,
            )
            with (
                patch.object(toolchains, "run") as run,
                patch.object(toolchains, "urlopen") as fetch,
            ):
                self.assertEqual(
                    toolchains.flatppl_compiler(flatppl, False, root), flatppl
                )
                self.assertEqual(
                    toolchains.iree_tools(args, root), (args.iree_dir, compiler)
                )
                run.assert_not_called()
                fetch.assert_not_called()

    def test_compiler_archive_checksum_and_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "compiler.whl"
            member = ZipInfo("iree/compiler/_mlir_libs/iree-compile" + toolchains.EXE)
            member.external_attr = 0o100755 << 16
            with ZipFile(archive, "w") as wheel:
                wheel.writestr(member, b"example executable")
            with self.assertRaisesRegex(toolchains.BuildError, "checksum mismatch"):
                toolchains.extract_compiler(archive, root / "bad", "0" * 64)
            self.assertFalse((root / "bad").exists())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            compiler = toolchains.extract_compiler(archive, root / "ok", digest)
            self.assertEqual(compiler.read_bytes(), b"example executable")
            self.assertTrue(os.access(compiler, os.X_OK))


if __name__ == "__main__":
    unittest.main()

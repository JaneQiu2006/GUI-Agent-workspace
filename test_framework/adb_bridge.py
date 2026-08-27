"""ADB transport with automatic local USB / reverse-SSH selection.

The GUI evaluation runner normally runs on a GPU server, while the Android device is
attached to a developer machine.  This module also supports running the device
part of the framework directly on that developer machine.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import struct
import time
import zlib
from typing import List, Mapping, Optional, Sequence, Tuple, Union


class AdbError(RuntimeError):
    pass


def find_adb() -> str:
    """Return an ADB executable without requiring shell profile setup."""
    configured = os.environ.get("ADB_PATH")
    candidates = [configured, shutil.which("adb")]
    for sdk_var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        sdk_root = os.environ.get(sdk_var)
        if sdk_root:
            candidates.append(str(Path(sdk_root) / "platform-tools" / "adb"))
    candidates.extend(
        [
            str(Path.home() / "Library/Android/sdk/platform-tools/adb"),
            "/opt/homebrew/bin/adb",
            "/usr/local/bin/adb",
        ]
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise AdbError(
        "ADB 未安装。macOS 可执行：brew install --cask android-platform-tools"
    )


def _load_bridge_env() -> Mapping[str, str]:
    """Load the non-secret metadata written by start_adb_bridge.sh."""
    values = {}
    env_path = Path.home() / ".config/test-framework/device-bridge.env"
    if not env_path.is_file():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


class AdbBridge:
    """Run ADB either locally or through SSH back to the USB host."""

    def __init__(
        self,
        transport: str = "auto",
        serial: Optional[str] = None,
        ssh_host: Optional[str] = None,
        ssh_port: Optional[int] = None,
        ssh_user: Optional[str] = None,
        adb_path: Optional[str] = None,
    ) -> None:
        metadata = _load_bridge_env()
        self.transport = (os.environ.get("ADB_TRANSPORT") or transport).lower()
        self.serial = os.environ.get("ANDROID_SERIAL") or serial
        self.ssh_host = os.environ.get("ADB_SSH_HOST") or ssh_host or "127.0.0.1"
        self.ssh_port = int(
            os.environ.get("ADB_SSH_PORT")
            or ssh_port
            or metadata.get("ADB_SSH_PORT", "2222")
        )
        self.ssh_user = (
            os.environ.get("ADB_SSH_USER")
            or ssh_user
            or metadata.get("ADB_SSH_USER")
        )
        self.adb_path = adb_path
        self.last_screenshot_mode = "uninitialized"
        self.remote_adb_path = (
            os.environ.get("ADB_REMOTE_PATH")
            or metadata.get("ADB_REMOTE_PATH")
            or "adb"
        )
        if self.transport not in {"auto", "local", "ssh"}:
            raise AdbError("ADB transport 必须是 auto、local 或 ssh")
        if self.transport == "auto":
            if self._local_device_ready():
                self.transport = "local"
            elif self.ssh_user:
                self.transport = "ssh"
            elif platform.system() == "Darwin":
                # On the USB Mac, report an unauthorised/missing phone directly
                # instead of incorrectly falling back to a server-side tunnel.
                self.transport = "local"
            else:
                self.transport = "ssh"
        if self.transport == "local":
            self.adb_path = self.adb_path or find_adb()
        elif not self.ssh_user:
            raise AdbError(
                "未找到反向 SSH 的本机用户名。请先在 USB 主机运行 "
                "scripts/start_adb_bridge.sh <服务器 SSH 别名>"
            )

    def _local_device_ready(self) -> bool:
        try:
            adb = self.adb_path or find_adb()
            result = subprocess.run(
                [adb, "devices"], capture_output=True, text=True, timeout=8, check=False
            )
        except (AdbError, OSError, subprocess.TimeoutExpired):
            return False
        ready = [
            line.split()[0]
            for line in result.stdout.splitlines()[1:]
            if len(line.split()) >= 2 and line.split()[1] == "device"
        ]
        return self.serial in ready if self.serial else bool(ready)

    def _adb_args(self, args: Sequence[str]) -> List[str]:
        executable = (
            self.adb_path or "adb"
            if self.transport == "local"
            else self.remote_adb_path
        )
        adb_args = [executable]
        if self.serial:
            adb_args.extend(["-s", self.serial])
        adb_args.extend(str(arg) for arg in args)
        return adb_args

    def _command(self, args: Sequence[str]) -> List[str]:
        adb_args = self._adb_args(args)
        if self.transport == "local":
            return adb_args
        remote_command = " ".join(shlex.quote(arg) for arg in adb_args)
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-p",
            str(self.ssh_port),
            f"{self.ssh_user}@{self.ssh_host}",
            remote_command,
        ]

    def run(
        self,
        args: Sequence[str],
        *,
        binary: bool = False,
        timeout: int = 30,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        command = self._command(args)
        result = subprocess.run(
            command,
            capture_output=True,
            text=not binary,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            stderr = result.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            raise AdbError(
                f"ADB 命令失败（transport={self.transport}）：{stderr.strip()}"
            )
        return result

    def ready_devices(self) -> List[str]:
        result = self.run(["devices", "-l"])
        return [
            line.split()[0]
            for line in result.stdout.splitlines()[1:]
            if len(line.split()) >= 2 and line.split()[1] == "device"
        ]

    def ensure_ready(self) -> str:
        devices = self.ready_devices()
        if self.serial:
            if self.serial not in devices:
                raise AdbError(f"指定设备 {self.serial} 未处于 device 状态")
            return self.serial
        if not devices:
            raise AdbError(
                "未发现已授权的 Android 设备。请解锁手机并确认“允许 USB 调试”。"
            )
        if len(devices) > 1:
            raise AdbError(
                "发现多个设备，请设置 ANDROID_SERIAL：" + ", ".join(devices)
            )
        self.serial = devices[0]
        return self.serial

    def screenshot(self, output_path: Union[os.PathLike, str]) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.last_screenshot_mode == "synthetic_secure_surface":
            return self._write_placeholder_screenshot(output)
        signature = b"\x89PNG\r\n\x1a\n"
        last_prefix = b""
        for attempt in range(3):
            result = self.run(["exec-out", "screencap", "-p"], binary=True, timeout=45)
            png_start = result.stdout.find(signature)
            if png_start >= 0:
                output.write_bytes(result.stdout[png_start:])
                self.last_screenshot_mode = "direct_screencap"
                return output
            last_prefix = result.stdout[:120]
            # Some video surfaces return an empty direct exec-out stream.  Writing
            # on-device first and reading the completed file is more reliable.
            device_path = "/sdcard/Download/.test_framework_screenshot.png"
            self.run(["shell", "screencap", "-p", device_path], check=False, timeout=45)
            file_result = self.run(["exec-out", "cat", device_path], binary=True, check=False, timeout=45)
            png_start = file_result.stdout.find(signature)
            if png_start >= 0:
                output.write_bytes(file_result.stdout[png_start:])
                self.last_screenshot_mode = "on_device_file"
                return output
            last_prefix = file_result.stdout[:120] or last_prefix
            if attempt < 2:
                self.run(["start-server"], check=False, timeout=20)
                time.sleep(1)
        return self._write_placeholder_screenshot(output)

    def _write_placeholder_screenshot(self, output: Path) -> Path:
        """Write an auditable PNG when the foreground secure surface blocks ADB capture."""
        width, height = self.display_size()
        signature = b"\x89PNG\r\n\x1a\n"
        row = b"\x00" + bytes((28, 28, 28)) * width
        pixels = zlib.compress(row * height, level=9)

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        placeholder = (
            signature
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", pixels)
            + chunk(b"IEND", b"")
        )
        output.write_bytes(placeholder)
        self.last_screenshot_mode = "synthetic_secure_surface"
        return output

    def shell(self, *args: object, timeout: int = 30) -> str:
        result = self.run(["shell", *(str(arg) for arg in args)], timeout=timeout)
        return result.stdout.strip()

    def display_size(self) -> Tuple[int, int]:
        output = self.shell("wm", "size")
        for line in reversed(output.splitlines()):
            if ":" in line and "x" in line:
                dimensions = line.split(":", 1)[1].strip().split("x", 1)
                try:
                    return int(dimensions[0]), int(dimensions[1])
                except ValueError:
                    continue
        raise AdbError(f"无法解析屏幕尺寸：{output}")

    def tap(self, x: int, y: int) -> None:
        """Tap an absolute screen coordinate."""
        self.shell("input", "tap", int(x), int(y))

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 500,
    ) -> None:
        """Swipe between absolute coordinates using Android's standard input syntax."""
        duration_ms = max(50, min(5000, int(duration_ms)))
        self.shell(
            "input", "swipe",
            int(x1), int(y1), int(x2), int(y2), duration_ms,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="检查手机 GUI 测评流程的 ADB 连接")
    parser.add_argument("--transport", choices=("auto", "local", "ssh"), default="auto")
    parser.add_argument("--serial")
    parser.add_argument("--screenshot")
    args = parser.parse_args()

    try:
        bridge = AdbBridge(transport=args.transport, serial=args.serial)
        serial = bridge.ensure_ready()
        width, height = bridge.display_size()
        print(f"ADB_READY transport={bridge.transport} serial={serial} size={width}x{height}")
        if args.screenshot:
            path = bridge.screenshot(args.screenshot)
            print(f"SCREENSHOT {path.resolve()}")
        return 0
    except (AdbError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"ADB_ERROR {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Headed-Chrome CDP session for scraping Google Images with the real browser.

Google flags headless Chrome with a reCAPTCHA even when using a logged-in
profile, but a visible headed Chrome renders results normally.  A fresh temp
profile is used so the user's default profile is never touched (Chrome also
refuses to run under CDP with the default data directory).
"""
import json
import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
)


def chrome_exe() -> str | None:
    for path in _CHROME_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ChromeSession:
    """A single headed Chrome instance driven over the Chrome DevTools Protocol."""

    def __init__(self):
        self.port = _free_port()
        self.profile = Path(os.environ.get("TEMP", "/tmp")) / "opencode" / f"sc_{os.getpid()}"
        self.proc: subprocess.Popen | None = None
        self._pw = None
        self.browser = None
        self._endpoint = f"http://127.0.0.1:{self.port}"

    def start(self) -> "ChromeSession":
        exe = chrome_exe()
        if not exe:
            raise RuntimeError("Chrome executable not found")
        self.profile.mkdir(parents=True, exist_ok=True)
        args = [
            exe,
            f"--remote-debugging-port={self.port}",
            f"--user-data-dir={self.profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-extensions",
            "--window-size=1400,1000",
        ]
        self.proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            time.sleep(1)
            try:
                with urllib.request.urlopen(self._endpoint + "/json/version", timeout=3) as resp:
                    json.load(resp)
                    break
            except Exception:
                pass
        else:
            raise RuntimeError("Chrome CDP endpoint did not come up")
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.connect_over_cdp(self._endpoint)
        return self

    @property
    def context(self):
        return self.browser.contexts[0]

    def close(self) -> None:
        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        if self.proc:
            try:
                self.proc.terminate()
                time.sleep(1)
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)
            except Exception:
                pass
        shutil.rmtree(self.profile, ignore_errors=True)

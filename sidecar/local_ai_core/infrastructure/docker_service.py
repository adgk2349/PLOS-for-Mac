from __future__ import annotations

import os
import subprocess
import threading
import logging

logger = logging.getLogger(__name__)

class DockerService:
    def __init__(self, compose_dir: str, idle_timeout_seconds: float | None = None):
        self.compose_dir = compose_dir
        self._lock = threading.Lock()
        self._idle_timer: threading.Timer | None = None
        self._idle_timeout_seconds = self._resolve_idle_timeout(idle_timeout_seconds)

    @staticmethod
    def _resolve_idle_timeout(value: float | None) -> float:
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return 600.0
        raw = str(os.getenv("LOCAL_AI_SEARXNG_IDLE_TIMEOUT_SECONDS", "600") or "").strip()
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 600.0

    def is_docker_available(self) -> bool:
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def is_desktop_running(self) -> bool:
        try:
            result = subprocess.run(["pgrep", "-x", "Docker"], capture_output=True, text=True)
            return bool(result.stdout.strip())
        except Exception:
            return False

    def is_running(self) -> bool:
        if not self.is_docker_available():
            return False
        try:
            result = subprocess.run(
                ["docker", "compose", "ps", "--status", "running", "--services"],
                cwd=self.compose_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return bool(result.stdout.strip())
        except Exception:
            return False

    def _cancel_idle_timer_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_idle_stop_locked(self) -> None:
        self._cancel_idle_timer_locked()
        if self._idle_timeout_seconds <= 0:
            return
        timer = threading.Timer(self._idle_timeout_seconds, self._idle_stop_worker)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_stop_worker(self) -> None:
        try:
            self.stop(shutdown_desktop=False, remove_stack=False)
        except Exception:
            logger.exception("Idle shutdown for SearXNG failed")

    def mark_usage(self, *, allow_auto_stop: bool = True) -> None:
        with self._lock:
            if allow_auto_stop:
                self._schedule_idle_stop_locked()
            else:
                self._cancel_idle_timer_locked()

    def start(self, *, keep_running: bool = False) -> bool:
        import time
        with self._lock:
            # 1. Start Docker Desktop app if not running
            if not self.is_desktop_running():
                logger.info("Starting Docker Desktop app...")
                subprocess.run(["open", "-a", "Docker"], check=False)
                
                # Wait for Docker binary to become available/ready
                max_retries = 30
                for i in range(max_retries):
                    if self.is_docker_available():
                        try:
                            # Check if the engine is actually responsive
                            subprocess.run(["docker", "info"], capture_output=True, check=True)
                            logger.info("Docker engine is ready.")
                            break
                        except Exception:
                            pass
                    time.sleep(2)
                    if i % 5 == 0:
                        logger.info(f"Waiting for Docker engine... ({i}/{max_retries})")
                else:
                    logger.error("Docker engine failed to start in time.")
                    return False
            
            if self.is_running():
                logger.info("SearXNG is already running")
                if keep_running:
                    self._cancel_idle_timer_locked()
                else:
                    self._schedule_idle_stop_locked()
                return True

            try:
                logger.info(f"Starting SearXNG in {self.compose_dir}")
                subprocess.run(
                    ["docker", "compose", "up", "-d"],
                    cwd=self.compose_dir,
                    check=True,
                    capture_output=True
                )
                if keep_running:
                    self._cancel_idle_timer_locked()
                else:
                    self._schedule_idle_stop_locked()
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to start SearXNG: {e.stderr.decode()}")
                return False

    def stop(self, *, shutdown_desktop: bool = False, remove_stack: bool = False) -> bool:
        with self._lock:
            self._cancel_idle_timer_locked()
            # 1. Stop containers
            try:
                command = ["docker", "compose", "down"] if remove_stack else ["docker", "compose", "stop"]
                subprocess.run(
                    command,
                    cwd=self.compose_dir,
                    check=False,
                    capture_output=True
                )
                logger.info("SearXNG containers stopped.")
            except Exception as e:
                logger.warning(f"Failed to stop SearXNG containers: {e}")

            # 2. Quit Docker app
            if shutdown_desktop and self.is_desktop_running():
                logger.info("Shutting down Docker Desktop app...")
                subprocess.run(["osascript", "-e", 'quit app "Docker"'], check=False)
                return True
            return True

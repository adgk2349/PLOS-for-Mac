import subprocess
from typing import Any
from .contracts import SystemToolProvider
from ..models import SystemFilePermission

class MacSystemTools(SystemToolProvider):
    def spotlight_search(self, query: str) -> list[str]:
        """Search for files using macOS spotlight (mdfind)."""
        try:
            result = subprocess.run(
                ["mdfind", query], 
                capture_output=True, 
                text=True, 
                check=True,
                timeout=5.0
            )
            paths = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return paths[:20]
        except Exception:
            return []

    def get_metadata(self, path: str) -> dict[str, Any]:
        """Fetch file metadata using mdls."""
        try:
            result = subprocess.run(
                ["mdls", path],
                capture_output=True,
                text=True,
                check=True,
                timeout=2.0
            )
            metadata = {}
            for line in result.stdout.splitlines():
                if " = " in line:
                    key, val = line.split(" = ", 1)
                    metadata[key.strip()] = val.strip()
            return metadata
        except Exception:
            return {}

    def execute_command(self, command: str, permission_level: SystemFilePermission) -> str:
        """Execute a general shell command with granular permission enforcement."""
        if permission_level == SystemFilePermission.FULL_ACCESS:
            # Dangerous, but allowed by user setting.
            pass
        else:
            read_only_cmds = {"ls", "grep", "cat", "head", "tail", "wc", "du", "df", "find", "whoami", "pwd"}
            modify_cmds = {"cp", "mv", "touch", "mkdir", "rm", "chmod"} 
            
            allowed = read_only_cmds
            if permission_level == SystemFilePermission.READ_WRITE:
                allowed = read_only_cmds | modify_cmds
            
            base_cmd = command.split()[0] if command.strip() else ""
            if base_cmd not in allowed:
                return (
                    f"Error: Command '{base_cmd}' is denied in current permission mode ({permission_level.value}). "
                    "You can upgrade permissions in Settings > System Permission Approval."
                )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10.0
            )
            return (result.stdout + result.stderr).strip() or "Command executed successfully (no output)."
        except Exception as e:
            return f"Error executing command: {str(e)}"

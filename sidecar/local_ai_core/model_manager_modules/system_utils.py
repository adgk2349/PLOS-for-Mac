from __future__ import annotations

import os


def system_memory_gb() -> int:
    override = (os.getenv("LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE") or "").strip()
    if override:
        try:
            parsed = int(float(override))
            if parsed > 0:
                return parsed
        except Exception:
            pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if isinstance(page_size, int) and isinstance(phys_pages, int) and page_size > 0 and phys_pages > 0:
            total_bytes = page_size * phys_pages
            return max(1, int(total_bytes / (1024**3)))
    except Exception:
        pass
    return 16


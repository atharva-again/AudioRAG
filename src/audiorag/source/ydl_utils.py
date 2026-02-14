from __future__ import annotations

from typing import Any

from audiorag.core.config import AudioRAGConfig

SUPPORTED_BROWSERS = frozenset(
    {
        "brave",
        "chrome",
        "chromium",
        "edge",
        "opera",
        "vivaldi",
        "whale",
        "firefox",
        "safari",
    }
)


def _validate_browser_string(browser: str) -> str:
    """Validate browser string format for yt-dlp cookiesfrombrowser.

    Args:
        browser: Browser string like "chrome", "firefox:default", "chrome+gnomekeyring:Profile1"

    Returns:
        The validated browser string

    Raises:
        ValueError: If browser name is not supported
    """
    parts = browser.split(":")
    browser_name = parts[0].split("+")[0]

    if browser_name not in SUPPORTED_BROWSERS:
        supported = ", ".join(sorted(SUPPORTED_BROWSERS))
        msg = f"Unsupported browser: {browser_name}. Supported: {supported}"
        raise ValueError(msg)

    return browser


def build_ydl_opts(config: AudioRAGConfig) -> dict[str, Any] | None:
    """Build yt-dlp options from config.

    Consolidates all YouTube-related yt-dlp configuration into a single
    options dict. Used by both pipeline and discovery modules.

    Args:
        config: AudioRAG configuration

    Returns:
        Dictionary of yt-dlp options, or None if no options configured
    """
    ydl_opts: dict[str, Any] = {}

    if config.youtube_cookie_file:
        ydl_opts["cookiefile"] = config.youtube_cookie_file

    if config.youtube_cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = _validate_browser_string(
            config.youtube_cookies_from_browser
        )

    if config.youtube_po_token:
        yt_args = ydl_opts.setdefault("extractor_args", {}).setdefault("youtube", {})
        token = config.youtube_po_token
        if "+" not in token:
            token = f"web.gvs+{token}"
        yt_args["po_token"] = [token]

    if config.youtube_visitor_data:
        ydl_opts.setdefault("extractor_args", {}).setdefault("youtube", {})["visitor_data"] = (
            config.youtube_visitor_data
        )

    if config.youtube_impersonate:
        from yt_dlp.networking.impersonate import ImpersonateTarget

        ydl_opts["impersonate"] = ImpersonateTarget.from_str(config.youtube_impersonate)

    return ydl_opts if ydl_opts else None

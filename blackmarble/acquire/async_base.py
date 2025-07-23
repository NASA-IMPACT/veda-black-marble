"""Base classes for async downloading with rate limiting and retry logic."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from aiohttp import ClientSession, ClientTimeout


class DownloadStatus(str, Enum):
    """Status of a download operation"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class DownloadResult:
    """Result of a download operation"""

    url: str
    local_path: Path
    status: DownloadStatus
    size_bytes: int | None = None
    duration_seconds: float | None = None
    error: str | None = None
    attempt_count: int = 1


@dataclass
class DownloadConfig:
    """Configuration for async downloads"""

    max_concurrent: int = 5
    timeout_seconds: int = 300
    chunk_size: int = 8192
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    headers: dict[str, str] = field(default_factory=dict)


class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Requests per second
            burst: Maximum burst size
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made"""
        async with self._lock:
            while self.tokens < 1:
                now = time.monotonic()
                elapsed = now - self.last_update
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now

                if self.tokens < 1:
                    sleep_time = (1 - self.tokens) / self.rate
                    await asyncio.sleep(sleep_time)

            self.tokens -= 1


class AsyncDownloaderBase(ABC):
    """Base class for async downloaders with common functionality"""

    def __init__(
        self,
        config: DownloadConfig | None = None,
        rate_limiter: RateLimiter | None = None,
        progress_callback: Callable[[DownloadResult], None] | None = None,
    ):
        """
        Args:
            config: Download configuration
            rate_limiter: Optional rate limiter
            progress_callback: Optional callback for progress updates
        """
        self.config = config or DownloadConfig()
        self.rate_limiter = rate_limiter
        self.progress_callback = progress_callback
        self.logger = logging.getLogger(self.__class__.__name__)

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

        # Session will be created in async context
        self._session: ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = ClientTimeout(total=self.config.timeout_seconds)
        self._session = ClientSession(timeout=timeout, headers=self.config.headers)
        return self

    async def __aexit__(self, _exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit"""
        if self._session:
            await self._session.close()

    @abstractmethod
    async def prepare_download(self, item: Any, output_dir: Path) -> tuple[str, Path]:
        """
        Prepare download URL and output path.

        Args:
            item: Item to download (source-specific)
            output_dir: Base output directory

        Returns:
            Tuple of (download_url, local_path)
        """

    async def download_file(self, url: str, output_path: Path, **kwargs: Any) -> DownloadResult:
        """
        Download a single file with retry logic.

        Args:
            url: URL to download
            output_path: Where to save the file
            **kwargs: Additional arguments for the request

        Returns:
            DownloadResult with status and metadata
        """
        result = DownloadResult(url=url, local_path=output_path, status=DownloadStatus.PENDING)

        # Check if file already exists
        if output_path.exists() and output_path.stat().st_size > 0:
            result.status = DownloadStatus.COMPLETED
            result.size_bytes = output_path.stat().st_size
            self.logger.info(f"File already exists: {output_path}")
            return result

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with retries
        start_time = time.time()

        for attempt in range(self.config.max_retries):
            try:
                result.attempt_count = attempt + 1
                result.status = DownloadStatus.IN_PROGRESS

                # Notify progress tracker that download is starting
                if self.progress_callback:
                    self.progress_callback(result)

                # Apply rate limiting if configured
                if self.rate_limiter:
                    await self.rate_limiter.acquire()

                # Download with semaphore for concurrency control
                async with self._semaphore:
                    await self._download_with_progress(url, output_path, result, **kwargs)

                result.status = DownloadStatus.COMPLETED
                result.duration_seconds = time.time() - start_time

                # Notify progress tracker that download completed
                if self.progress_callback:
                    self.progress_callback(result)

                return result

            except Exception as e:
                self.logger.warning(f"Download attempt {attempt + 1} failed for {url}: {e}")
                result.error = str(e)

                if attempt < self.config.max_retries - 1:
                    result.status = DownloadStatus.RETRYING
                    delay = self.config.retry_delay * (self.config.retry_backoff**attempt)
                    await asyncio.sleep(delay)
                else:
                    result.status = DownloadStatus.FAILED
                    if self.progress_callback:
                        self.progress_callback(result)
                    raise

        return result

    async def _download_with_progress(
        self, url: str, output_path: Path, result: DownloadResult, **kwargs: Any
    ) -> None:
        """Download file with progress tracking"""
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")

        # Use temporary file for atomic writes
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        try:
            async with self._session.get(url, **kwargs) as response:
                response.raise_for_status()

                # Check content type to detect HTML error pages
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" in content_type:
                    # Read first 1000 chars of HTML to get error details
                    html_content = await response.text()
                    html_preview = html_content[:500]
                    raise RuntimeError(
                        f"Received HTML error page instead of expected file. "
                        f"This usually indicates an authentication issue. "
                        f"Content preview: {html_preview}..."
                    )

                # Get total size if available
                total_size = int(response.headers.get("content-length", 0))
                result.size_bytes = total_size

                # Download in chunks
                downloaded = 0
                first_chunk = True
                with open(temp_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(self.config.chunk_size):
                        # Extra check on first chunk for HTML content
                        if first_chunk:
                            if chunk.startswith(b"<!DOCTYPE") or chunk.startswith(b"<html"):
                                temp_path.unlink()  # Clean up
                                raise RuntimeError(
                                    f"Downloaded file appears to be HTML, not expected file type. "
                                    f"This usually indicates an authentication or access issue. "
                                    f"First bytes: {chunk[:100]}"
                                )
                            first_chunk = False
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Progress callback could be enhanced here
                        # to report partial progress

            # Atomic rename
            temp_path.rename(output_path)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def download_batch(self, items: list[Any], output_dir: Path) -> list[DownloadResult]:
        """
        Download multiple items concurrently.

        Args:
            items: List of items to download
            output_dir: Base output directory

        Returns:
            List of DownloadResults
        """
        tasks = []

        for item in items:
            url, output_path = await self.prepare_download(item, output_dir)
            task = self.download_file(url, output_path)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                url, output_path = await self.prepare_download(items[i], output_dir)
                final_results.append(
                    DownloadResult(
                        url=url,
                        local_path=output_path,
                        status=DownloadStatus.FAILED,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results


class DownloadProgressTracker:
    """Track progress across multiple downloads using logging"""

    def __init__(self, total_files: int, description: str = "Downloading", parent_task=None):
        """
        Args:
            total_files: Total number of files to download
            description: Description for the progress tracking
            parent_task: Optional parent task for nested progress (future use)
        """
        self.total = total_files
        self.completed = 0
        self.failed = 0
        self.in_progress = 0
        self.description = description
        self.parent_task = parent_task
        self.last_log_percent = -1
        
        # Log initial status
        logging.info("Starting %s: %d files", self.description, self.total)

    def update(self, result: DownloadResult):
        """Update progress based on download result"""
        if result.status == DownloadStatus.IN_PROGRESS:
            self.in_progress += 1
        elif result.status == DownloadStatus.COMPLETED:
            self.completed += 1
            self.in_progress = max(0, self.in_progress - 1)
        elif result.status == DownloadStatus.FAILED:
            self.failed += 1
            self.in_progress = max(0, self.in_progress - 1)
            logging.warning("Download failed: %s", result.url)

        # Log progress at 10% intervals
        if self.total > 0:
            current_percent = int((self.completed + self.failed) * 100 / self.total)
            if current_percent >= self.last_log_percent + 10:
                self.last_log_percent = current_percent
                logging.info("%s progress: %d%% (%d/%d completed, %d failed, %d in progress)",
                             self.description, current_percent, self.completed, self.total, 
                             self.failed, self.in_progress)

    def finish(self):
        """Log final summary"""
        # Log summary
        status = "✅" if self.failed == 0 else "⚠️"
        logging.info("%s %s complete: %d succeeded, %d failed",
                     status, self.description, self.completed, self.failed)

"""
Read URL Tool - Fetches and extracts text content from URLs.
Supports HTML pages and PDF documents.
"""
import logging
import re
import socket
from typing import Dict, Any, Optional
from ipaddress import ip_address
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class ReadUrlTool:
    """Fetch and extract text content from URLs."""

    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_CHARS = 100000
    DEFAULT_MAX_BYTES = 5_000_000
    MAX_REDIRECTS = 5
    USER_AGENT = "Mozilla/5.0 (compatible; StockResearchBot/1.0)"

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_bytes: int = DEFAULT_MAX_BYTES
    ):
        self.timeout = timeout
        self.max_chars = max_chars
        self.max_bytes = max_bytes
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                }
            )
        return self._client

    def _is_blocked_host(self, hostname: str) -> bool:
        """Block private, loopback, or otherwise reserved hosts."""
        if not hostname:
            return True
        normalized = hostname.strip().lower().rstrip(".")
        if normalized in {"localhost"}:
            return True
        try:
            ip = ip_address(normalized)
            return (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_multicast
                or ip.is_reserved
                or ip.is_unspecified
            )
        except ValueError:
            # Resolve hostname and block if any IP is private/reserved
            try:
                infos = socket.getaddrinfo(normalized, None)
            except Exception:
                return True
            for info in infos:
                ip_str = info[4][0]
                try:
                    ip = ip_address(ip_str)
                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_multicast
                        or ip.is_reserved
                        or ip.is_unspecified
                    ):
                        return True
                except ValueError:
                    continue
        return False

    def _validate_url(self, url: str) -> Optional[str]:
        """Validate URL format and block unsafe hosts."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Unsupported URL scheme"
        if not parsed.netloc or not parsed.hostname:
            return "Invalid URL format"
        if self._is_blocked_host(parsed.hostname):
            return "Blocked host"
        return None

    async def _read_limited(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read response body with a hard byte limit."""
        chunks = []
        total = 0
        truncated = False
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > self.max_bytes:
                allowed = self.max_bytes - (total - len(chunk))
                if allowed > 0:
                    chunks.append(chunk[:allowed])
                truncated = True
                break
            chunks.append(chunk)
        return b"".join(chunks), truncated

    async def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Fetch URL with validation, redirect checks, and size limits."""
        client = await self._get_client()
        current_url = url
        redirects = 0

        while True:
            error = self._validate_url(current_url)
            if error:
                return {"success": False, "error": error, "url": current_url}

            async with client.stream("GET", current_url) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        return {"success": False, "error": "Redirect with no Location header", "url": current_url}
                    redirects += 1
                    if redirects > self.MAX_REDIRECTS:
                        return {"success": False, "error": "Too many redirects", "url": current_url}
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > self.max_bytes:
                            return {"success": False, "error": "Content too large", "url": current_url}
                    except ValueError:
                        pass

                content_bytes, truncated = await self._read_limited(response)
                return {
                    "success": True,
                    "url": current_url,
                    "content_bytes": content_bytes,
                    "content_type": response.headers.get("content-type", "").lower(),
                    "encoding": response.encoding,
                    "truncated_bytes": truncated,
                }

    async def read_url(
        self,
        url: str,
        extract_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Fetch and extract text content from a URL.

        Args:
            url: URL to fetch
            extract_type: "auto", "html", or "pdf"

        Returns:
            Dict with extracted content and metadata
        """
        try:
            fetch_result = await self._fetch_url(url)
            if not fetch_result.get("success"):
                return fetch_result

            final_url = fetch_result["url"]
            content_type = fetch_result.get("content_type", "")
            content_bytes = fetch_result.get("content_bytes", b"")
            encoding = fetch_result.get("encoding")
            truncated_bytes = fetch_result.get("truncated_bytes", False)

            # Determine extraction method
            if extract_type == "auto":
                if "pdf" in content_type or final_url.lower().endswith(".pdf"):
                    extract_type = "pdf"
                else:
                    extract_type = "html"

            # Extract text based on type
            if extract_type == "pdf":
                if truncated_bytes:
                    return {
                        "success": False,
                        "error": "PDF exceeds maximum size",
                        "url": final_url
                    }
                text = await self._extract_pdf(content_bytes)
                title = final_url
            else:
                html_text = self._decode_html(content_bytes, encoding)
                text = self._extract_html(html_text)
                title = self._extract_title(html_text)

            # Truncate if too long
            truncated = truncated_bytes
            if len(text) > self.max_chars:
                text = text[:self.max_chars] + "\n\n[Content truncated...]"
                truncated = True

            return {
                "success": True,
                "url": final_url,
                "title": title,
                "content": text,
                "content_type": extract_type,
                "char_count": len(text),
                "truncated": truncated,
            }

        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching URL: {url}")
            return {
                "success": False,
                "error": "Request timed out",
                "url": url
            }
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error fetching URL {url}: {e.response.status_code}")
            return {
                "success": False,
                "error": f"HTTP {e.response.status_code}",
                "url": url
            }
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }

    def _decode_html(self, content_bytes: bytes, encoding: Optional[str]) -> str:
        """Decode HTML bytes with a safe fallback."""
        if encoding:
            try:
                return content_bytes.decode(encoding, errors="replace")
            except Exception:
                pass
        try:
            return content_bytes.decode("utf-8", errors="replace")
        except Exception:
            return content_bytes.decode("latin-1", errors="replace")

    def _extract_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove unwanted elements
            for element in soup.find_all([
                "script", "style", "nav", "footer", "header",
                "aside", "form", "button", "iframe", "noscript"
            ]):
                element.decompose()

            # Try to find main content area
            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.find(class_=re.compile(r"(content|article|post|entry)", re.I)) or
                soup.find(id=re.compile(r"(content|article|post|entry)", re.I)) or
                soup.body or
                soup
            )

            # Get text with some structure preservation
            text = self._get_text_with_structure(main_content)

            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' {2,}', ' ', text)

            return text.strip()

        except Exception as e:
            logger.error(f"Error extracting HTML: {e}")
            # Fallback to basic text extraction
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n", strip=True)

    def _get_text_with_structure(self, element) -> str:
        """Extract text preserving some structure (headers, paragraphs)."""
        if element is None:
            return ""

        lines = []

        for child in element.descendants:
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                text = child.get_text(strip=True)
                if text:
                    lines.append(f"\n## {text}\n")
            elif child.name == 'p':
                text = child.get_text(strip=True)
                if text:
                    lines.append(f"{text}\n")
            elif child.name == 'li':
                text = child.get_text(strip=True)
                if text:
                    lines.append(f"- {text}")
            elif child.name == 'br':
                lines.append("\n")

        if not lines:
            return element.get_text(separator="\n", strip=True)

        return "\n".join(lines)

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        try:
            soup = BeautifulSoup(html, "html.parser")
            # Try og:title first (often better)
            og_title = soup.find("meta", property="og:title")
            if og_title and og_title.get("content"):
                return og_title["content"]
            # Fall back to title tag
            if soup.title and soup.title.string:
                return soup.title.string.strip()
            # Try h1
            h1 = soup.find("h1")
            if h1:
                return h1.get_text(strip=True)
            return ""
        except Exception:
            return ""

    async def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            # Use pdfplumber if available (already in requirements)
            import pdfplumber
            import io

            text_parts = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages[:50]:  # Limit to first 50 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            return "\n\n".join(text_parts)

        except ImportError:
            logger.warning("pdfplumber not available, using pypdf fallback")
            try:
                from pypdf import PdfReader
                import io

                reader = PdfReader(io.BytesIO(content))
                text_parts = []
                for page in reader.pages[:50]:
                    text_parts.append(page.extract_text() or "")
                return "\n\n".join(text_parts)

            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return "[Unable to extract PDF content]"

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return "[Unable to extract PDF content]"

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Tool definition for Groq API
READ_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "read_url",
        "description": "Fetch and extract text content from a URL (HTML page or PDF). Use this after web_search to get the full content of a promising result.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch and extract content from"
                },
                "extract_type": {
                    "type": "string",
                    "enum": ["auto", "html", "pdf"],
                    "description": "Content extraction type. Use 'auto' to detect automatically.",
                    "default": "auto"
                }
            },
            "required": ["url"]
        }
    }
}

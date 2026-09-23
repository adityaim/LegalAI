"""
modules/response/__init__.py
──────────────────────────────
Public API surface for Module 7: Response Generation & Formatting.

Usage::

    from modules.response import ResponseFormatter, FormattedResponse

    formatter = ResponseFormatter()
    response = formatter.format(legal_answer, fmt="markdown")

    print(response.to_markdown())   # for UI rendering
    print(response.to_plain())      # for logging / plain text clients
    print(response.to_html())       # for web embedding
    print(response.to_json_summary())  # for API inspection
"""

from .formatter import ResponseFormatter, FormatterConfig
from .schema import FormattedResponse, CitationBlock, ResponseFormat

__all__ = [
    "ResponseFormatter",
    "FormatterConfig",
    "FormattedResponse",
    "CitationBlock",
    "ResponseFormat",
]

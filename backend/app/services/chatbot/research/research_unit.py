"""
Research Unit - Single research unit with LLM-driven tool loop.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, AsyncGenerator

from sqlalchemy.orm import Session

from ....config import settings
from ....services.llm import LLMService, LLMError, LLMRateLimitError, LLMContextWindowError
from .config import research_settings
from .models import SubQuestion, SourceNote, ResearchUnitResult, ThinkResult, ThinkDecision
from .prompts import RESEARCH_UNIT_SYSTEM_PROMPT, RESEARCH_UNIT_USER_PROMPT
from ..tool_definitions import (
    WEB_SEARCH_TOOL,
    SEARCH_NEWS_TOOL,
    SEARCH_FINANCE_TOOL,
    GET_SEC_10K_TOOL,
    READ_IR_PDF_TOOL,
    RESEARCH_THEME_TOOL,
    DISCOVER_THEMES_TOOL,
)
from ..tools.read_url import READ_URL_TOOL, ReadUrlTool
from ..tools.web_search import WebSearchTool
from ..tools.database_tools import DatabaseTools

logger = logging.getLogger(__name__)


# Tool definitions for research unit (subset of available tools)
SUMMARIZE_SOURCE_TOOL = {
    "type": "function",
    "function": {
        "name": "summarize_source",
        "description": "Create structured research notes from source content. Call after reading a URL or document to extract key facts.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to summarize (from read_url or document tool)"
                },
                "source_url": {
                    "type": "string",
                    "description": "URL of the source for citation"
                },
                "source_title": {
                    "type": "string",
                    "description": "Title of the source"
                },
                "research_question": {
                    "type": "string",
                    "description": "The research question this should address"
                }
            },
            "required": ["content", "source_url", "source_title", "research_question"]
        }
    }
}

THINK_TOOL = {
    "type": "function",
    "function": {
        "name": "think",
        "description": "Reasoning checkpoint. Reflect on your research progress and decide whether to continue searching, try a different approach, or stop with sufficient data.",
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Your reasoning about the current research state"
                },
                "decision": {
                    "type": "string",
                    "enum": ["continue_research", "sufficient_data", "need_different_angle"],
                    "description": "What to do next"
                },
                "next_query": {
                    "type": ["string", "null"],
                    "description": "Next search query if continuing research (null if stopping)"
                }
            },
            "required": ["thought", "decision"]
        }
    }
}


class ResearchUnit:
    """Single research unit that investigates one sub-question."""

    # Tools available to research units
    RESEARCH_TOOLS = [
        WEB_SEARCH_TOOL,
        SEARCH_NEWS_TOOL,
        SEARCH_FINANCE_TOOL,
        READ_URL_TOOL,
        SUMMARIZE_SOURCE_TOOL,
        THINK_TOOL,
        GET_SEC_10K_TOOL,
        READ_IR_PDF_TOOL,
        RESEARCH_THEME_TOOL,
        DISCOVER_THEMES_TOOL,
    ]

    def __init__(
        self,
        db: Session,
        sub_question: SubQuestion,
        unit_index: int = 0,
        max_tool_calls: Optional[int] = None
    ):
        self.db = db
        self.sub_question = sub_question
        self.unit_index = unit_index
        self.max_tool_calls = max_tool_calls or research_settings.deep_research_max_tool_calls_per_unit
        self.notes: List[SourceNote] = []
        self.tool_calls_count = 0
        # Track sources from search results (fallback if summarize_source not called)
        self._captured_urls: set = set()
        self._search_sources: List[Dict[str, Any]] = []
        self.llm = LLMService(use_case="research")
        self._init_tools()
        # Database tools for theme research
        self.database_tools = DatabaseTools(db)

    def _init_tools(self):
        """Initialize tool executors."""
        self.web_search = WebSearchTool()
        self.read_url_tool = ReadUrlTool(
            timeout=research_settings.read_url_timeout,
            max_chars=research_settings.read_url_max_chars,
            max_bytes=research_settings.read_url_max_bytes
        )

    async def execute(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the research loop for this sub-question.

        Yields:
            Progress events (dicts) and final ResearchUnitResult
        """
        # Build initial messages
        search_queries_text = "\n".join([f"- {q}" for q in self.sub_question.search_queries])
        user_prompt = RESEARCH_UNIT_USER_PROMPT.format(
            question=self.sub_question.question,
            search_queries=search_queries_text or "- " + self.sub_question.question
        )

        messages = [
            {"role": "system", "content": RESEARCH_UNIT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        max_iterations = research_settings.deep_research_max_iterations

        for iteration in range(max_iterations):
            if self.tool_calls_count >= self.max_tool_calls:
                logger.info(f"Unit {self.unit_index}: Hit tool call limit ({self.max_tool_calls})")
                break

            # Call LLM with tools
            try:
                response = await self._call_llm_with_tools(messages)
            except LLMContextWindowError as e:
                logger.warning(f"Unit {self.unit_index} request too large after truncation: {e}")
                yield {
                    "type": "research_unit_error",
                    "unit": self.unit_index,
                    "error": f"Request too large: {e}"
                }
                break
            except LLMRateLimitError as e:
                logger.error(f"Unit {self.unit_index} rate limit exhausted: {e}")
                yield {
                    "type": "research_unit_error",
                    "unit": self.unit_index,
                    "error": f"Rate limit exhausted: {e}"
                }
                break
            except LLMError as e:
                logger.error(f"Unit {self.unit_index} LLM error: {e}")
                yield {
                    "type": "research_unit_error",
                    "unit": self.unit_index,
                    "error": str(e)
                }
                break
            except Exception as e:
                logger.error(f"Unit {self.unit_index} unexpected error: {e}")
                yield {
                    "type": "research_unit_error",
                    "unit": self.unit_index,
                    "error": str(e)
                }
                break

            assistant_message = response.choices[0].message

            # Check if LLM wants to call tools
            if not assistant_message.tool_calls:
                # No more tool calls - done researching
                logger.info(f"Unit {self.unit_index}: LLM finished (no tool calls)")
                break

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })

            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                self.tool_calls_count += 1

                # Yield progress event
                yield {
                    "type": "research_unit_tool",
                    "unit": self.unit_index,
                    "tool": tool_name,
                    "iteration": iteration
                }

                # Execute the tool
                result = await self._execute_tool(tool_name, tool_args)

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(result, default=str)
                })

                # Handle special tool results
                if tool_name == "summarize_source" and result.get("note"):
                    self.notes.append(result["note"])
                    yield {
                        "type": "research_unit_note",
                        "unit": self.unit_index,
                        "source": result["note"].source_title,
                        "notes_count": len(self.notes)
                    }

                # Auto-capture references from search tools as fallback sources
                # This ensures we have sources even if LLM doesn't call summarize_source
                if tool_name in ("web_search", "search_news", "search_finance"):
                    search_refs = result.get("references", [])
                    for ref in search_refs:
                        if ref.get("url") and ref["url"] not in self._captured_urls:
                            self._captured_urls.add(ref["url"])
                            self._search_sources.append({
                                "url": ref.get("url", ""),
                                "title": ref.get("title", "Search Result"),
                                "type": ref.get("type", "web"),
                                "snippet": ref.get("snippet", "")
                            })

                if tool_name == "think":
                    decision = result.get("decision")
                    if decision == "sufficient_data":
                        logger.info(f"Unit {self.unit_index}: Sufficient data collected")
                        # Clean up and yield final result
                        await self._cleanup()
                        # Use fallback sources if no explicit notes
                        final_notes = self.notes
                        if not final_notes and self._search_sources:
                            logger.info(f"Unit {self.unit_index}: Using {len(self._search_sources)} search sources as fallback")
                            for src in self._search_sources[:10]:
                                final_notes.append(SourceNote(
                                    source_url=src["url"],
                                    source_title=src["title"],
                                    source_type=src["type"],
                                    content_summary=src.get("snippet", "Source from search results"),
                                    key_facts=[],
                                    relevance_score=0.5,
                                    research_question=self.sub_question.question
                                ))
                        yield ResearchUnitResult(
                            sub_question=self.sub_question.question,
                            notes=final_notes,
                            iterations=iteration + 1,
                            completed=True,
                            tool_calls_made=self.tool_calls_count
                        )
                        return

        # Clean up
        await self._cleanup()

        # If no notes from summarize_source, create fallback notes from search sources
        final_notes = self.notes
        if not final_notes and self._search_sources:
            logger.info(f"Unit {self.unit_index}: No summarize_source notes, using {len(self._search_sources)} search sources as fallback")
            for src in self._search_sources[:10]:  # Limit to 10 fallback sources
                final_notes.append(SourceNote(
                    source_url=src["url"],
                    source_title=src["title"],
                    source_type=src["type"],
                    content_summary=src.get("snippet", "Source from search results"),
                    key_facts=[],
                    relevance_score=0.5,
                    research_question=self.sub_question.question
                ))

        # Yield final result
        yield ResearchUnitResult(
            sub_question=self.sub_question.question,
            notes=final_notes,
            iterations=iteration + 1 if 'iteration' in locals() else 0,
            completed=len(final_notes) > 0,
            tool_calls_made=self.tool_calls_count
        )

    async def _call_llm_with_tools(self, messages: List[Dict], retry_with_truncation: bool = True) -> Any:
        """Call LLM with tools using LLMService."""
        # Format messages for API
        formatted = self._format_messages_for_api(messages)

        try:
            return await self.llm.completion(
                messages=formatted,
                tools=self.RESEARCH_TOOLS,
                tool_choice="auto",
                max_tokens=4000,
                num_retries=research_settings.groq_max_retries,
            )
        except LLMContextWindowError as e:
            if retry_with_truncation and len(messages) > 2:
                # Try truncating tool results to reduce size
                logger.warning(f"Request too large, truncating messages: {e}")
                truncated = self._truncate_messages(messages)
                return await self._call_llm_with_tools(truncated, retry_with_truncation=False)
            raise

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Truncate tool result content to reduce message size."""
        truncated = []
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                # Truncate large tool results
                if len(content) > 2000:
                    truncated_content = content[:2000] + "\n...[truncated due to size]"
                    truncated.append({**msg, "content": truncated_content})
                else:
                    truncated.append(msg)
            else:
                truncated.append(msg)
        return truncated

    def _format_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for LLM API."""
        formatted = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                formatted.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": msg["tool_calls"]
                })
            elif msg.get("role") == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg["tool_call_id"],
                    "name": msg.get("name", ""),
                    "content": msg["content"]
                })
            else:
                formatted.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        return formatted

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        try:
            if tool_name == "web_search":
                return await self.web_search.search(
                    query=args.get("query", ""),
                    max_results=args.get("max_results", 5),
                    search_type="general"
                )

            elif tool_name == "search_news":
                return await self.web_search.search_news(
                    query=args.get("query", ""),
                    max_results=args.get("max_results", 5)
                )

            elif tool_name == "search_finance":
                return await self.web_search.search_finance(
                    query=args.get("query", ""),
                    max_results=args.get("max_results", 5)
                )

            elif tool_name == "read_url":
                return await self.read_url_tool.read_url(
                    url=args.get("url", ""),
                    extract_type=args.get("extract_type", "auto")
                )

            elif tool_name == "summarize_source":
                return self._summarize_source(args)

            elif tool_name == "think":
                return self._process_think(args)

            elif tool_name == "get_sec_10k":
                # Import here to avoid circular imports
                from ..tools.document_tools import DocumentTools
                doc_tools = DocumentTools(self.db)
                return await doc_tools.get_sec_10k(
                    symbol=args.get("symbol", ""),
                    year=args.get("year"),
                    query=args.get("query")
                )

            elif tool_name == "read_ir_pdf":
                from ..tools.document_tools import DocumentTools
                doc_tools = DocumentTools(self.db)
                return await doc_tools.read_ir_pdf(
                    url=args.get("url", ""),
                    query=args.get("query")
                )

            elif tool_name == "research_theme":
                result = self.database_tools.research_theme(
                    theme_name=args.get("theme_name", ""),
                    include_sources=args.get("include_sources", True),
                    include_history=args.get("include_history", False),
                    max_sources=args.get("max_sources", 10),
                    max_constituents=args.get("max_constituents", 20)
                )
                if result is None:
                    return {"error": f"Theme '{args.get('theme_name', '')}' not found"}
                return result

            elif tool_name == "discover_themes":
                return self.database_tools.discover_themes(
                    mode=args.get("mode", "trending"),
                    theme_names=args.get("theme_names"),
                    min_velocity=args.get("min_velocity", 1.0),
                    category=args.get("category"),
                    limit=args.get("limit", 10)
                )

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"error": str(e)}

    def _summarize_source(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process summarize_source tool call - creates a SourceNote."""
        content = args.get("content", "")
        source_url = args.get("source_url", "")
        source_title = args.get("source_title", "")
        research_question = args.get("research_question", self.sub_question.question)

        # Extract key facts using a simple heuristic
        # In production, you might use another LLM call here
        sentences = content.split(". ")
        key_facts = []
        for sentence in sentences[:10]:  # First 10 sentences
            sentence = sentence.strip()
            if len(sentence) > 30 and len(sentence) < 300:
                # Look for sentences with numbers or key terms
                if any(c.isdigit() for c in sentence) or \
                   any(term in sentence.lower() for term in ["million", "billion", "percent", "%", "growth", "revenue"]):
                    key_facts.append(sentence)

        # Create content summary (first 500 chars)
        content_summary = content[:500] + "..." if len(content) > 500 else content

        note = SourceNote(
            source_url=source_url,
            source_title=source_title,
            source_type=self._infer_source_type(source_url),
            content_summary=content_summary,
            key_facts=key_facts[:5],  # Max 5 key facts
            relevance_score=0.8,
            research_question=research_question
        )

        return {
            "success": True,
            "note": note,
            "facts_extracted": len(key_facts)
        }

    def _process_think(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Process think tool call - reasoning checkpoint."""
        thought = args.get("thought", "")
        decision = args.get("decision", "continue_research")
        next_query = args.get("next_query")

        logger.info(f"Unit {self.unit_index} think: {decision} - {thought[:100]}")

        return {
            "thought": thought,
            "decision": decision,
            "next_query": next_query,
            "notes_collected": len(self.notes),
            "status": "acknowledged"
        }

    def _infer_source_type(self, url: str) -> str:
        """Infer source type from URL."""
        url_lower = url.lower()
        if "sec.gov" in url_lower:
            return "sec"
        elif ".pdf" in url_lower:
            return "pdf"
        elif any(domain in url_lower for domain in ["reuters", "bloomberg", "wsj", "cnbc", "yahoo.com/news"]):
            return "news"
        else:
            return "web"

    async def _cleanup(self):
        """Clean up resources."""
        await self.web_search.close()
        await self.read_url_tool.close()

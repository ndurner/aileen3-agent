from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any

from gradio import ChatMessage


@dataclass
class ToolDisplayState:
    """
    Per-turn state for displaying tool usage as Gradio 'thought' messages.

    - tools_by_key: canonical id -> ChatMessage (one per logical tool run)
    - tool_order: stable order in which tools were first seen
    - job_roots: job_id -> canonical id (for start_*/get_* long-running tools)
    """

    tools_by_key: dict[str, ChatMessage] = field(default_factory=dict)
    tool_order: list[str] = field(default_factory=list)
    job_roots: dict[str, str] = field(default_factory=dict)
    # Track how often we've seen an in-progress update for a given tool key.
    fake_progress_ticks: dict[str, int] = field(default_factory=dict)
    # When true, suppress subsequent assistant text chunks for this turn.
    suppress_text: bool = False


def init_tool_display_state() -> ToolDisplayState:
    return ToolDisplayState()


def _first_part(event: dict) -> dict[str, Any]:
    content = event.get("content") or {}
    parts = content.get("parts") or []
    if not parts:
        return {}
    return parts[0] or {}


def _extract_function_call(part: dict) -> dict | None:
    # Support both snake_case and camelCase keys.
    return part.get("function_call") or part.get("functionCall")


def _extract_function_response(part: dict) -> dict | None:
    # Support both snake_case and camelCase keys.
    return part.get("function_response") or part.get("functionResponse")


def _pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except TypeError:
        return str(obj)


def _snake_to_title(name: str) -> str:
    return " ".join(chunk.capitalize() for chunk in name.split("_") if chunk)


def _tool_label(name: str) -> tuple[str, str]:
    """
    Return (emoji, human_label) for a tool function name.
    """
    mapping: dict[str, tuple[str, str]] = {
        "get_factual_memory": ("📚", "Memory lookup"),
        "start_media_retrieval": ("📺", "Load media"),
        "start_media_analysis": ("🔍", "Analyze media"),
        "get_media_analysis_result": ("🔍", "Analyze media"),
    }
    if name in mapping:
        return mapping[name]
    # Default: generic wrench + title-cased name.
    return "🛠️", _snake_to_title(name)


def _format_args_markdown(args: dict[str, Any]) -> str:
    if not args:
        return "_No arguments._"
    lines = ["**Input**"]
    for key, value in args.items():
        pretty_value = _pretty_json(value) if isinstance(value, (dict, list)) else str(value)
        lines.append(f"- **{key}**: {pretty_value}")
    return "\n".join(lines)


def _format_media_analysis_args(args: dict[str, Any]) -> str:
    """
    Pretty formatter for start_media_analysis arguments, especially the priors object.
    """
    reference = args.get("reference")
    priors = args.get("priors") or {}

    lines: list[str] = []
    if reference:
        lines.append(f"**Media reference:** `{reference}`")

    if isinstance(priors, dict) and priors:
        def add_field(key: str, label: str) -> None:
            value = priors.get(key)
            if not value:
                return
            text = str(value).strip()
            if not text:
                return
            lines.append("")
            lines.append(f"**{label}**")
            lines.append("")
            lines.append(text)

        add_field("context", "Context")
        add_field("expectations", "Expectations")
        add_field("prior_knowledge", "Prior knowledge")
        add_field("questions", "Questions")
    elif not lines:
        lines.append("_No briefing details provided._")

    return "\n".join(lines) if lines else "_No input provided._"


def _extract_structured(response: dict) -> dict | None:
    """
    Get structuredContent from a tool response, if present.
    """
    if not isinstance(response, dict):
        return None
    structured = response.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    return None


def _parse_text_content(response: dict) -> str | None:
    """
    Extract a single text blob from response['content'][...]['text'], if present.
    """
    if not isinstance(response, dict):
        return None
    content_items = response.get("content") or []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "text":
            continue
        text_value = item.get("text") or ""
        if text_value:
            return text_value
    return None


def _strip_simple_xml(text: str) -> str:
    """
    Best-effort stripping of a single <tag ...>value</tag> wrapper.
    Used for get_factual_memory's <memory>...</memory> payloads.
    """
    start = text.find(">")
    end = text.rfind("<")
    if start != -1 and end != -1 and end > start:
        inner = text[start + 1 : end].strip()
        return inner or text
    return text


def _format_get_factual_memory_body(response: dict) -> str:
    """
    Turn the memory tool result into friendly markdown.
    """
    result = response.get("result")
    if isinstance(result, str):
        cleaned = _strip_simple_xml(result)
        return f"**Memory lookup result**\n\n{cleaned}"
    # Fallback to JSON view.
    return f"**Memory lookup result**\n\n```json\n{_pretty_json(response)}\n```"


def _format_media_retrieval_body(structured: dict) -> str:
    reference = structured.get("reference")
    status_label = structured.get("status")
    cached = structured.get("cached")
    metadata_block = structured.get("metadata", {}) or {}

    title = metadata_block.get("title")
    source = metadata_block.get("source") or structured.get("source")
    duration = metadata_block.get("duration")
    channel = metadata_block.get("channel")
    description = metadata_block.get("description")

    lines: list[str] = []
    if title:
        lines.append(f"**Title:** {title}")
    if reference:
        lines.append(f"**Reference:** `{reference}`")
    if source:
        lines.append(f"**Source:** {source}")
    if duration is not None:
        lines.append(f"**Duration:** {duration} seconds")
    if channel:
        lines.append(f"**Channel:** {channel}")
    if status_label is not None:
        extra = f", cached: {cached}" if cached is not None else ""
        lines.append(f"**Status:** `{status_label}`{extra}")
    if description:
        lines.append("")
        lines.append(description)
    return "\n".join(lines) if lines else "_No media details available._"


def _data_uri_from(value: Any) -> str | None:
    """
    Convert raw slide/image representations into a data URI string.

    Mirrors the behavior of the slide_utils helper in the core repo so that
    the chat UI can consume the same tool outputs.
    """
    if not value:
        return None

    if isinstance(value, str):
        if value.startswith("data:"):
            return value
        return None

    if isinstance(value, bytes):
        b64 = base64.b64encode(value).decode("ascii")
        return f"data:image/png;base64,{b64}"

    # Handle ImageContent-like objects or dicts
    data = getattr(value, "data", None)
    mime = getattr(value, "mimeType", None) or getattr(value, "mime_type", None)

    if isinstance(value, dict):
        data = value.get("data", data)
        mime = value.get("mimeType") or value.get("mime_type") or mime

    if data is None:
        return None

    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    else:
        if isinstance(data, str) and data.startswith("data:"):
            return data
        if not isinstance(data, str):
            return None
        b64 = data

    if not mime:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


def _normalize_slide_entries(slides_result: Any) -> list[dict]:
    """
    Return a list of slide dicts with `image_data_uri` populated.

    This is adapted from the core demo's slide_utils.normalize_slide_entries
    so the chat UI can render slide images directly.
    """
    if not isinstance(slides_result, dict):
        return []

    candidates: Any = slides_result.get("slides")
    if isinstance(candidates, dict):
        candidates = candidates.get("slides")
    elif candidates is None:
        maybe_nested = slides_result.get("result")
        if isinstance(maybe_nested, dict):
            candidates = maybe_nested.get("slides")
            if isinstance(candidates, dict):
                candidates = candidates.get("slides")

    if not isinstance(candidates, list):
        return []

    normalized: list[dict] = []
    for idx, raw in enumerate(candidates):
        entry: dict | None = None
        if isinstance(raw, dict):
            entry = dict(raw)
            data_uri = entry.get("image_data_uri")
            if not data_uri:
                for key in ("image", "image_content", "content"):
                    if key in entry:
                        data_uri = _data_uri_from(entry[key])
                        if data_uri:
                            entry["image_data_uri"] = data_uri
                            break
            entry.setdefault("index", idx)
        else:
            data_uri = _data_uri_from(raw)
            if data_uri:
                entry = {"index": idx, "image_data_uri": data_uri}

        if entry and entry.get("image_data_uri"):
            normalized.append(entry)

    return normalized


def _image_md_from_data_uri(data_uri: str, alt: str) -> str | None:
    """
    Build an HTML image tag from a data URI.

    Using HTML instead of markdown allows us to control the display size
    inside the Gradio Chatbot while still working with render_markdown=True.
    """
    if not isinstance(data_uri, str):
        return None
    if not data_uri.startswith("data:"):
        return None
    # Use a reasonably large width while keeping responsiveness.
    return f'<img src="{data_uri}" alt="{alt}" style="max-width: 100%; width: 768px; height: auto;" />'


def _fake_progress_bar(ticks: int) -> str:
    """
    Return a playful, fake progress indicator based on the number of
    intermediate updates we've seen for a tool.
    """
    # Grows over time; purely for entertainment.
    blocks = "▁▂▃▄▅▆▇█"
    length = min(max(ticks, 1), len(blocks))
    bar = blocks[:length]
    return f"{bar} (thinking…) "


def _format_media_analysis_body(structured: dict, progress_ticks: int | None = None) -> str:
    """
    Format media_analysis structuredContent as end-user friendly markdown, omitting
    low-level ids and internals.
    """
    status_label = structured.get("status")
    analysis = structured.get("analysis")

    lines: list[str] = []

    if isinstance(analysis, dict):
        # Final, rich analysis payload.
        title = analysis.get("title")
        slide_count = analysis.get("slide_count")
        source = analysis.get("source")

        if title:
            lines.append(f"**Title:** {title}")
        if slide_count is not None:
            lines.append(f"**Slides:** {slide_count}")
        if source:
            lines.append(f"**Source:** {source}")

        analysis_text = analysis.get("analysis")
        if analysis_text:
            if lines:
                lines.append("")
            # The analysis body is already markdown.
            lines.append(analysis_text)
    else:
        # Intermediate / running payload: no full analysis yet.
        reference = structured.get("reference")

        if status_label:
            lines.append(f"**Status:** `{status_label}`")
        if reference:
            lines.append(f"**Reference:** `{reference}`")

        if progress_ticks is not None:
            fake_bar = _fake_progress_bar(progress_ticks)
            lines.append(f"**Progress:** {fake_bar}")
        else:
            lines.append("_Media analysis in progress..._")

    if status_label and status_label != "done" and not any(
        l.startswith("**Status:**") for l in lines
    ):
        # Prepend status badge if still running and not already included.
        lines.insert(0, f"**Status:** `{status_label}`")

    return "\n".join(lines)


def process_event(
    event: dict,
    state: ToolDisplayState,
) -> str | None:
    """
    Update the per-turn tool display state based on a raw ADK / API server event.

    Returns:
        A text chunk (for the main assistant response) if this event carries
        user-visible text, otherwise None.
    """
    author = event.get("author")

    # --- Briefing refinement agent events ---------------------------------
    # We treat these as a synthetic tool call so the UI can show a spinner
    # while the refinement is running, without streaming its raw text.
    if author == "briefing_refinement_agent":
        key = "__briefing_refinement__"
        msg = state.tools_by_key.get(key)
        emoji = "👩🏻‍🏫"
        label = "Refining and expanding inquiry"
        title = f"{emoji} {label}"

        if msg is None:
            msg = ChatMessage(
                role="assistant",
                content="_Refining and expanding your inquiry..._",
                metadata={
                    "title": title,
                    "status": "pending",
                    "id": key,
                },
            )
            state.tools_by_key[key] = msg
            state.tool_order.append(key)
        else:
            if msg.metadata is None:
                msg.metadata = {}
            msg.metadata["title"] = title

        # Mark as done when the refinement agent signals completion. ADK
        # events typically carry finishReason when a stream ends; fall back
        # to non-partial events as a completion signal.
        finish_reason = event.get("finishReason")
        partial = event.get("partial")
        if finish_reason == "STOP" or (partial is None or partial is False):
            msg.metadata["status"] = "done"
        else:
            msg.metadata["status"] = "pending"

        # Never surface the refinement agent's own text chunks.
        return None

    part = _first_part(event)
    if not part:
        return None

    # --- Tool call events -------------------------------------------------
    fc = _extract_function_call(part)
    if fc:
        tool_name = fc.get("name", "tool")
        args = fc.get("args", {}) or {}
        call_id = fc.get("id") or tool_name

        emoji, label = _tool_label(tool_name)
        title = f"{emoji} {label}"

        if tool_name == "start_media_analysis":
            body = _format_media_analysis_args(args)
        else:
            body = _format_args_markdown(args)

        msg = state.tools_by_key.get(call_id)
        if msg is None:
            msg = ChatMessage(
                role="assistant",
                content=body,
                metadata={
                    "title": title,
                    "status": "pending",
                    "id": call_id,
                },
            )
            state.tools_by_key[call_id] = msg
            state.tool_order.append(call_id)
        else:
            # Update args body if we somehow see multiple calls with same id.
            msg.content = body
            if msg.metadata is not None:
                msg.metadata["title"] = title
                msg.metadata["status"] = "pending"

        # No user-visible text chunk; the thought message itself is streamed
        # via ChatMessage.
        return None

    # --- Tool response events ---------------------------------------------
    fr = _extract_function_response(part)
    if fr:
        tool_name = fr.get("name", "tool")
        response = fr.get("response", {}) or {}
        call_id = fr.get("id") or tool_name

        emoji, label = _tool_label(tool_name)
        title = f"{emoji} {label}"

        structured = _extract_structured(response) or {}
        job_id = structured.get("job_id") if isinstance(structured, dict) else None
        status_value = structured.get("status") if isinstance(structured, dict) else None

        # Work out canonical tool key so that start_* / get_* share one message.
        canonical_key = call_id
        if job_id:
            root = state.job_roots.get(job_id)
            if root is None:
                state.job_roots[job_id] = call_id
            else:
                canonical_key = root

        msg = state.tools_by_key.get(canonical_key)
        if msg is None:
            # We missed the functionCall; create the message on first response.
            msg = ChatMessage(
                role="assistant",
                content="",
                metadata={
                    "title": title,
                    "id": canonical_key,
                },
            )
            state.tools_by_key[canonical_key] = msg
            state.tool_order.append(canonical_key)

        # Update metadata.
        if msg.metadata is None:
            msg.metadata = {}
        msg.metadata["title"] = title
        if status_value:
            msg.metadata["status"] = "done" if status_value == "done" else "pending"
        else:
            msg.metadata["status"] = "done"

        # Build a nice body depending on the tool.
        if tool_name == "get_factual_memory":
            msg.content = _format_get_factual_memory_body(response)
        elif tool_name in ("start_media_retrieval", "get_media_retrieval_status"):
            msg.content = _format_media_retrieval_body(structured)
        elif tool_name in ("start_media_analysis", "get_media_analysis_result"):
            # Increment fake progress ticks for long-running media analysis jobs
            # so each poll animates the playful progress bar.
            if status_value and status_value != "done":
                state.fake_progress_ticks[canonical_key] = state.fake_progress_ticks.get(canonical_key, 0) + 1
            ticks = state.fake_progress_ticks.get(canonical_key)
            msg.content = _format_media_analysis_body(structured, progress_ticks=ticks)
        elif tool_name in ("start_slide_extraction", "get_extracted_slides"):
            # Summarize slide extraction in the tool message, but surface the
            # actual slide images in the main assistant response instead of
            # inside the thought.
            slides = _normalize_slide_entries(structured or response)
            if slides:
                msg.content = f"Extracted {len(slides)} slides for this media."
                # Emit images into the regular assistant turn.
                lines: list[str] = [f"Detected {len(slides)} slides for this media."]
                for slide in slides:
                    uri = slide.get("image_data_uri")
                    if not uri:
                        continue
                    index = slide.get("index")
                    label = (slide.get("label") or "").strip()
                    start = slide.get("from")
                    end = slide.get("to")
                    time_range = ""
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        time_range = f"{int(start)}s–{int(end)}s"
                    caption_parts = []
                    if index is not None:
                        caption_parts.append(f"Slide #{index}")
                    if label:
                        caption_parts.append(label)
                    if time_range:
                        caption_parts.append(time_range)
                    caption = " · ".join(caption_parts)

                    img_md = _image_md_from_data_uri(uri, alt=f"Slide {index}")
                    if img_md:
                        lines.append("")
                        lines.append(img_md)
                    if caption:
                        lines.append("")
                        lines.append(caption)

                # After returning the images, suppress any noisy assistant text
                # the model might produce for this turn.
                state.suppress_text = True
                return "\n".join(lines)
            else:
                msg.content = "_No slides available for this media._"
        elif tool_name == "translate_slide":
            # Translate slide returns an ImageContent-like payload. Surface the
            # translated image in the main assistant response and keep the tool
            # message itself concise.
            data_uri: str | None = None
            if structured:
                data_uri = _data_uri_from(structured)
            if not data_uri and isinstance(response, dict):
                content_items = response.get("content") or []
                for item in content_items:
                    if isinstance(item, dict) and item.get("type") == "image":
                        data_uri = _data_uri_from(item)
                        if data_uri:
                            break

            if data_uri:
                img_md = _image_md_from_data_uri(data_uri, alt="Translated slide")
                msg.content = "Translated slide image ready."
                state.suppress_text = True
                return img_md or "_Translated slide image available._"
            else:
                msg.content = "_Slide translation completed, but the image payload could not be decoded._"
        else:
            # Generic fallback for other tools: pretty-print payload but avoid
            # overwhelming users with raw internals when possible.
            inner = _parse_text_content(response)
            if inner:
                try:
                    parsed = json.loads(inner)
                    body = f"```json\n{_pretty_json(parsed)}\n```"
                except Exception:
                    body = inner
            else:
                body = f"```json\n{_pretty_json(response)}\n```"

            msg.content = body

        return None

    # --- Plain text streaming chunks --------------------------------------
    if "text" in part:
        text = part["text"]
        if state.suppress_text:
            # Ignore chatter after image-producing tools for this turn.
            return None
        return text

    return None


def get_ordered_tool_messages(state: ToolDisplayState) -> list[ChatMessage]:
    """
    Return tool messages in first-seen order for display.
    """
    return [state.tools_by_key[k] for k in state.tool_order if k in state.tools_by_key]

import sys

from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

from .conditional_prep_agent import ConditionalPrepAgent
from .get_factual_memory_tool import get_factual_memory_tool

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

AILEEN3_MCP_TOOL_NAMES = [
    "start_media_retrieval",
    "get_media_retrieval_status",
    "start_media_analysis",
    "get_media_analysis_result",
    "start_media_transcription",
    "get_media_transcription_result",
    "search_youtube"
]

aileen3_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=["-m", "aileen3_mcp.server"],
        ),
        timeout=30.0,
    ),
    tool_filter=AILEEN3_MCP_TOOL_NAMES,
)

assi_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="assistant_agent",
    description="A helpful assistant for user questions named `Aileen`.",
    instruction="""You are Aileen, an expectation driven briefing assistant for long form talks,
webinars and conference sessions.

Your job is to help the user forage for signal in noisy content:
- highlight surprises relative to what the user already expects or knows
- point out new actors, artefacts and dependencies that receive attention
- answer the user's concrete questions using the analyzed media and factual memory.

You have three main sources of prior information:
1) The original user briefing.
2) The refined briefing produced once per session by [briefing_refinement_agent].
3) Factual memories from the Vertex AI Memory Bank via the `get_factual_memory` tool.

Treat the Expectations and Prior knowledge blocks as the user's baseline script.
Do not re-explain this baseline in detail unless the user asks.
Focus instead on deviations from it and on concrete, decision relevant details.

When a media_url is present and the user asks you to analyze, summarize or brief them
on that media, you should:
    - ensure the media has been retrieved via the aileen3 MCP tools
    (`start_media_retrieval` and `get_media_retrieval_status`)
    - trigger `start_media_analysis` with a `priors` object built from the refined briefing
    and any high confidence factual memories
    - use `get_media_analysis_result` to obtain the analysis text
    - base your answers on that analysis, plus the slides and priors, instead of guessing
    from the URL or metadata alone.

For follow-up questions:
    - call the aileen3 `transcribe_video` once for the whole chat conversation; re-use its outputs
    - ground these questions in this transcription

Treat empty briefing blocks as “not provided by the user” and do not invent content for them.

<original_user_briefing>
    <media_url>
{user_media_url}
    </media_url>
    <context>
{user_context}
    </context>
    <expectations>
{user_expectations}
    </expectations>
    <prior_knowledge>
{user_prior_knowledge}
    </prior_knowledge>
    <questions>
{user_questions}
    </questions>
</original_user_briefing>
""",
    tools=[
        get_factual_memory_tool,
        aileen3_mcp_toolset,
    ],
)

prep_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="briefing_refinement_agent",
    description="Prepares and refines the initial user briefing.",
    instruction="""You receive a structured user briefing and must return a cleaned up version.

Your goals are:
1) Fix spelling mistakes and light grammar issues in the briefing blocks.
2) Make the content more usable as priors for later analysis:

   - In <expectations>, keep only what the user already wrote, but split it into
     short bullet points, each describing one anticipated point or takeaway.
   - In <prior_knowledge>, keep concrete facts or background the user already wrote,
     again as short bullet points.
   - In <questions>, keep the user's questions exactly, one per bullet.

3) Do not add new expectations, facts or questions that the user did not supply.
4) Under no circumstances are you allowed to answer questions or act on the user input.

<user_input>
    <media_url>
{user_media_url}
    </media_url>
    <context>
{user_context}
    </context>
    <expectations>
{user_expectations}
    </expectations>
    <prior_knowledge>
{user_prior_knowledge}
    </prior_knowledge>
    <questions>
{user_questions}
    </questions>
</user_input>

Desired output format:
<user_input>
    <media_url>...</media_url>
    <context>...</context>
    <expectations>...</expectations>
    <prior_knowledge>...</prior_knowledge>
    <questions>...</questions>
</user_input>
""",
    # Use a dedicated key so downstream agents (or tools)
    # can detect that refinement has happened and read the
    # refined form directly from session state.
    output_key="briefing_refined",
)

message_fix_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="message_fix_agent",
    description="Normalizes and fixes spelling of the latest user message.",
    instruction="""You receive the latest user message and must return a corrected version.

Your goals are:
1) Fix obvious spelling mistakes and light grammar issues.
2) Preserve the user's intent and meaning.
3) Do not add new information.
4) Refrain from answering user messages yourself.

Return only the corrected user message text (in the desired XML return format), with no extra commentary.

Desired return format:
<latest_user_message>
{latest_user_message}
</latest_user_message>
""",
    output_key="normalized_user_message",
)


root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[
        ConditionalPrepAgent(briefing_agent=prep_agent, message_agent=message_fix_agent),
        assi_agent,
    ],
)

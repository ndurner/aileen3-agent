import sys

from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.agents.loop_agent import LoopAgent
from google.adk.agents.sequential_agent import SequentialAgent

from .assistant_loop_exit_agent import AssistantLoopExitAgent
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
        timeout=1200.0,
    ),
    tool_filter=AILEEN3_MCP_TOOL_NAMES,
)

assi_agent = Agent(
    model="gemini-3-pro-preview",
    name="assistant_agent",
    description="A helpful assistant for user questions named `Aileen`.",
    instruction="""You are Aileen, an expectation driven briefing assistant for long form talks,
webinars and conference sessions.

Consistent with the concept of Information Foraging, your job is to help the user forage for signal in noisy content:
- highlight surprises relative to what the user already expects or knows
- point out new actors, artefacts and dependencies that receive attention
- answer the user's concrete questions using the analyzed media and factual memory.

The "noisy content" is supplied to you by the user, e.g. through the media_url.

You have three main sources of prior information, in addition to the noisy content:
1) The original user briefing.
2) The refined briefing produced once per session by [briefing_refinement_agent].
3) Factual memories from the Vertex AI Memory Bank via the `get_factual_memory` function.

Treat the Expectations and Prior knowledge blocks as the user's baseline script.
Do not re-explain this baseline in detail unless the user asks.
Focus instead on deviations from it and on concrete, decision relevant details.

When a media_url is present and the user asks you to analyze, summarize or brief them
on that media, you should:
    1. retrieve the media via the `start_media_retrieval` tool
    2. ensure that retrieval is complete by using the `get_media_retrieval_status` tool repeatedly until the status is confirmed as 'done'
    3. use the `start_media_analysis` tool with a `priors` object built from the refined briefing and any high confidence factual memories
    4. use `get_media_analysis_result` to obtain the analysis text (call repeatedly if status is not 'done')
    5. formulate a response to the user. Base the response on the analysis from step #4, plus the slides and priors. Refrain from guessing from the URL or metadata alone.

For follow-up questions submitted by the user:
    1. call the `transcribe_video` tool once for the whole chat conversation; re-use its outputs
    2. formulate a response grounded this transcription

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

If you encounter an error while calling the tools or functions, report it to the user.
""",
    tools=[
        get_factual_memory_tool,
        aileen3_mcp_toolset,
    ],
    output_key="assistant_agent_response",
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

assistant_loop_exit_agent = AssistantLoopExitAgent(
    watched_agent_name=assi_agent.name,
    response_state_key="assistant_agent_response",
)

assistant_loop = LoopAgent(
    name="assistant_agent_loop",
    sub_agents=[
        assi_agent,
        assistant_loop_exit_agent,
    ],
    max_iterations=8,
)

root_agent = SequentialAgent(
    name="root_agent",
    sub_agents=[
        ConditionalPrepAgent(briefing_agent=prep_agent, message_agent=message_fix_agent),
        assistant_loop,
    ],
)

from google.adk.agents.llm_agent import Agent, LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent

from .conditional_prep_agent import ConditionalPrepAgent


assi_agent = Agent(
    model="gemini-2.5-flash-lite",
    name="assistant_agent",
    description="A helpful assistant for user questions named `Aileen`.",
    instruction="""Answer user questions to the best of your knowledge.

Use the briefing from the user and the refinements made to the same briefing by the [briefing_refinement_agent] to ground names, context, expectations, and questions. Treat empty blocks as not provided by the user.

<original_user_briefing>
    <media_url>
{user_media_url}
    </media_url>
    <context>
{user_context}
    </context>
    <expectation>
{user_expectations}
    <expectation>
    <prior_knowledge>
{user_prior_knowledge}
    </prior_knowledge>
    <questions>
{user_questions}
    </questions>
</original_user_briefing>
""",
)


prep_agent = LlmAgent(
    model="gemini-2.5-flash-lite",
    name="briefing_refinement_agent",
    description="Prepares and refines the initial user briefing.",
    instruction="""Your goals are:
1) Fix any spelling mistakes in the user_input.
2) Return your results in the same format.

<user_input>
    <media_url>
{user_media_url}
    </media_url>
    <context>
{user_context}
    </context>
    <expectation>
{user_expectations}
    <expectation>
    <prior_knowledge>
{user_prior_knowledge}
    </prior_knowledge>
    <questions>
{user_questions}
    </questions>
</user_input>

Desired output format:
<user_input>
    ...
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

<latest_user_message>
{latest_user_message}
</latest_user_message>

Return only the corrected user message text, with no extra commentary.
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

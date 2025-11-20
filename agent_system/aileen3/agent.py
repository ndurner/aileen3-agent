from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent

assi_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='assi_agent',
    description='A helpful assistant for user questions named `Aileen`.',
    instruction="""Answer user questions to the best of your knowledge.

Use the user briefing below to ground names, context, expectations, and questions. Treat empty blocks as not provided by the user.

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

root_agent = SequentialAgent(
    name = 'root_agent',
    sub_agents= [assi_agent]
)

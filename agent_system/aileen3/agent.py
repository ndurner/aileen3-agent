from google.adk.agents.llm_agent import Agent
from google.adk.agents.sequential_agent import SequentialAgent

assi_agent = Agent(
    model='gemini-2.5-flash-lite',
    name='assi_agent',
    description='A helpful assistant for user questions named `Aileen`.',
    instruction="""Answer user questions to the best of your knowledge.

Use the user briefing below to ground names, context, expectations, and questions. Treat empty blocks as not provided by the user.

~~~ User input
```Media URL
{user_media_url}
```
```Context
{user_context}
```
```Expectations
{user_expectations}
```
```Prior Knowledge
{user_prior_knowledge}
```
```Questions
{user_questions}
```
~~~
""",
)

root_agent = SequentialAgent(
    name = 'root_agent',
    sub_agents= [assi_agent]
)

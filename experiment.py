import importlib
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import tool
import tool_utils
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import InvalidToolCall



# 1. Create the model and memory:
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key="")
memory = ConversationBufferMemory(return_messages=True)

tools = dir(tool_utils)[8:]
tool_lib = importlib.import_module("tool_utils")
tools = [tool(getattr(tool_lib,t)) for t in tools]
tools_cfg = [convert_to_openai_tool(t) for t in tools]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 3. Create the Prompt:
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, you are provided with a list of tools use that to answer user's query. Execute the tools and use the data to answer user's query",
        ),
        # This is where the agent will write/read its messages from
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}"),
    ]
)

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

# 4. Formats the python function tools into JSON schema and binds them to the model:
llm_with_tools = model.bind_tools(tools=[convert_to_openai_tool(t) for t in tools])

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

# Construct the OpenAI functions agent:
agent = create_openai_tools_agent(llm_with_tools, tools, prompt)

# Create an agent executor by passing in the agent, tools and memory:
memory = ConversationBufferMemory(return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

agent_executor.invoke({"input": "What's going on with TSLA stock also tell me the holding details of user: 69123 in equities."})


# Striming data
import time
def streaming_function():
    # Your original function logic
    output1 = "out1" 
    time.sleep(2) # Calculate output 1
    yield output1
    
    output2 = "out2"
    time.sleep(3)  # Calculate output 2
    yield output2
    
    output3 = "out3"
    time.sleep(4)  # Calculate output 3
    yield output3
    
    output4 = "out4"
    time.sleep(2)  # Calculate output 4
    yield output4

# Example usage
stream = streaming_function()
for output in stream:
    print(output)  # Process each output one by one

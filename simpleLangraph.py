from langgraph.graph import Graph
from langgraph.prebuilt import chat_agent_executor
from langchain_core.messages import HumanMessage
from llm import OCILLM

llm = OCILLM()

def user_input(state):
    print("User Input Node ", state)
    return {"messages": [HumanMessage(content=state["input"])]}

def chatbot_response(state):
    response = llm.llm.invoke(state["messages"])
    return {"messages": response}

# Build graph
workflow = Graph()
workflow.add_node("user_input", user_input)
workflow.add_node("chatbot", chatbot_response)
workflow.add_edge("user_input", "chatbot")
workflow.set_entry_point("user_input")
workflow.set_finish_point("chatbot")

app = workflow.compile()

# Run
result = app.invoke({"input": "Hello, how are you?"})
print(result["messages"].content)


from langgraph.graph import Graph
import networkx as nx
import matplotlib.pyplot as plt

# Define node functions
def start_node(state):
    print("Starting workflow...")
    return {"input": state}

def process_node(state):
    print(f"Processing: {state['input']}")
    processed_data = str(state["input"]).upper()
    return {"processed": processed_data}

def end_node(state):
    print(f"Completed: {state['processed']}")
    return {"result": state["processed"]}


# Create and configure graph
workflow = Graph()
workflow.add_node("start", start_node)
workflow.add_node("process", process_node)
workflow.add_node("end", end_node)
workflow.add_edge("start", "process")
workflow.add_edge("process", "end")
workflow.set_entry_point("start")
workflow.set_finish_point("end")

# Visualize
G = nx.DiGraph()
G.add_nodes_from(["start", "process", "end"])
G.add_edges_from([("start", "process"), ("process", "end")])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightgreen",
        font_size=12, font_weight="bold", arrowsize=30)
plt.title("LangGraph Workflow Visualization")
plt.savefig("workflow_diagram.png",
           dpi=300,
           bbox_inches="tight",
           transparent=False)

# Run the workflow
app = workflow.compile()
inputs = {"input": "hello world"}

result = app.invoke(inputs)
print("Execution result:", result["result"])

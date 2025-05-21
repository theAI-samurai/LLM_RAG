from langgraph.graph import Graph
import networkx as nx
import matplotlib.pyplot as plt

# Define node functions
def input_node(state):
    try:
        x = float(state["input"])
        return {"number": x, "original": x}
    except (ValueError, KeyError):
        raise ValueError("Input must be a number")

def check_even_node(state):
    x = state["number"]
    if x % 2 == 0:
        flag = 'even'
    else:
        flag = 'odd'
    return {"path": flag, "number": x, "original":state["original"]}

def even_operation_node(state):
    x = state["number"]
    result = x / 2
    return {"result": result, "operation": "divided by 2", "original":state["original"]}

def odd_operation_node(state):
    x = state["number"]
    result = x / 10
    return {"result": result, "operation": "divided by 10", "original":state["original"]}

def output_node(state):
    return {
        "original": state["original"],
        "result": state["result"],
        "operation": state["operation"]
    }

# Create and configure graph
workflow = Graph()

# Add nodes
workflow.add_node("input", input_node)
workflow.add_node("check_even", check_even_node)
workflow.add_node("even_op", even_operation_node)
workflow.add_node("odd_op", odd_operation_node)
workflow.add_node("output", output_node)

# Add edges
workflow.add_edge("input", "check_even")
workflow.add_conditional_edges(
    "check_even",
    lambda state: state["path"],
    {
        "even": "even_op",
        "odd": "odd_op"
    }
)
workflow.add_edge("even_op", "output")
workflow.add_edge("odd_op", "output")

workflow.set_entry_point("input")
workflow.set_finish_point("output")

# Create visualization
G = nx.DiGraph()
G.add_nodes_from(["input", "check_even", "even_op", "odd_op", "output"])
G.add_edges_from([
    ("input", "check_even"),
    ("check_even", "even_op"),
    ("check_even", "odd_op"),
    ("even_op", "output"),
    ("odd_op", "output")
])

# Draw the graph
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        arrowsize=20,
        edge_color="gray")

# Add edge labels for conditions
edge_labels = {
    ("check_even", "even_op"): "Even",
    ("check_even", "odd_op"): "Odd"
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("Number Processing Workflow")
plt.savefig("number_workflow.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graph saved as 'number_workflow.png'")

# Run the workflow
app = workflow.compile()

# Test cases
test_numbers = [4, 7, 10, 15]
for num in test_numbers:
    result = app.invoke({"input": num})
    print(f"\nInput: {result['original']}")
    print(f"Operation: {result['operation']}")
    print(f"Result: {result['result']}")
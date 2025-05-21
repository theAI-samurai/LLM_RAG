from langgraph.graph import Graph
import networkx as nx
import matplotlib.pyplot as plt

# Node Definitions

def input_status_node(state):
    try:
        x = state["input"]
        return {"current_status": x, "original_status": x}
    except (ValueError, KeyError):
        raise ValueError("Input must be Offiline/")

def offline_operation(state):
    print('Are all the Power cables seated at both ends?\n And are the PSUs completely seated and present?')
    inp = input('Enter YES / NO to above question')
    return {"current_status": 'PWD/PSU OK!',
            "original_status": state['original_status'],
            "path" : inp
            }

def offline_yes_led(state):
    print("Are the front / back LED's showing any light?")
    inp = input('Enter YES / NO to above question')
    return {"current_status": 'Front / Back LEDs OK!',
            "original_status": state['original_status'],
            "path": inp
            }

def offline_no_led(state):              # Must go back to offline operation
    print("After insuring all PWD / PSU Device should power ON")
    return {"current_status": "Reverted to Offline operation Node",
            "original_status": state['original_status'],
            "path" : ''
            }

def yes_led_yes(state):
    print("Ask Engineer to consol device and troubleshoot.\n Device or component must be replaced ")
    return {"current_status": 'OK, End! Recommedation done!',
            "original_status": state['original_status'],
            "path" : ''
            }

def yes_led_no(state):
    print('Do any other devices connected to same PDU have power?')
    inp = input('Enter YES / NO to above question')
    return {"current_status": 'OK! PWD check same PDU',
            "original_status": state['original_status'],
            "path": inp
            }

def same_pdu_yes(state):
    print("Inform Engineering, 'other device connected have PWR' PDU could be faulty. Switch Router!")
    return {"current_status": 'OK! END, Recommedation done! ',
            "original_status": state['original_status'],
            "path": ''
            }

def same_pdu_no(state):
    print("Inform Engineering, 'other device connected do not have PWR' PDU isn't powered.")
    return {"current_status": 'OK! END, Recommedation done! ',
            "original_status": state['original_status'],
            "path": ''
            }


# Create and configure graph
workflow = Graph()

# Add nodes
workflow.add_node("input_offline", input_status_node)
workflow.add_node("check_PWD_cables", offline_operation)
workflow.add_node("check_LED", offline_yes_led)
workflow.add_node("offline_PWD_no", offline_no_led)
workflow.add_node("check_LED_yes", yes_led_yes)
workflow.add_node("check_PDU", yes_led_no)
workflow.add_node("check_PDU_yes", same_pdu_yes)
workflow.add_node("check_PDU_no", same_pdu_no)

# Add edges
workflow.add_edge("input_offline", "check_PWD_cables")
workflow.add_conditional_edges("check_PWD_cables", lambda state:state['path'], {'YES':"check_LED", 'NO':"offline_PWD_no"})
workflow.add_edge("offline_PWD_no", "check_PWD_cables")
workflow.add_conditional_edges("check_LED",  lambda state:state['path'], {'YES':'check_LED_yes', 'NO':'check_PDU'})
workflow.add_conditional_edges("check_PDU",  lambda state:state['path'], {'YES':'check_PDU_yes', 'NO':'check_PDU_no'})

workflow.set_entry_point("input_offline")
workflow.set_finish_point("check_LED_yes")
workflow.set_finish_point("check_PDU_yes")
workflow.set_finish_point("check_PDU_no")

# Create visualization
G = nx.DiGraph()
G.add_nodes_from(["input_offline", "check_PWD_cables", "check_LED", "offline_PWD_no",
                 "check_LED_yes", "check_PDU", "check_PDU_yes", "check_PDU_no"])

G.add_edges_from([
    ("input_offline", "check_PWD_cables"),
    ("check_PWD_cables", "check_LED"),
    ("check_PWD_cables", "offline_PWD_no"),
    ("offline_PWD_no", "check_PWD_cables"),
    ("check_LED", "check_LED_yes"),
    ("check_LED", "check_PDU"),
    ("check_PDU", "check_PDU_yes"),
    ("check_PDU", "check_PDU_no")
])

# Draw the graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        arrowsize=20,
        edge_color="gray")

# Highlight end nodes
end_nodes = ["check_LED_yes", "check_PDU_yes", "check_PDU_no"]
nx.draw_networkx_nodes(G, pos, nodelist=end_nodes, node_color="lightgreen", node_size=3000)

# Add edge labels
edge_labels = {
    ("check_PWD_cables", "check_LED"): "YES",
    ("check_PWD_cables", "offline_PWD_no"): "NO",
    ("check_LED", "check_LED_yes"): "YES",
    ("check_LED", "check_PDU"): "NO",
    ("check_PDU", "check_PDU_yes"): "YES",
    ("check_PDU", "check_PDU_no"): "NO"
}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

plt.title("Network Troubleshooting Workflow", pad=20)
plt.savefig("network_troubleshooting_workflow.png", dpi=300, bbox_inches="tight")
plt.close()

print("Graph saved as 'network_troubleshooting_workflow.png'")

# Run the workflow
app = workflow.compile()
result = app.invoke({"input": "Offline"})
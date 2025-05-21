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
workflow.add_node("input offline", input_status_node)
workflow.add_node("check_PWD_cables", offline_operation)
workflow.add_node("check_LED", offline_yes_led)
workflow.add_node("offline_PWD_no", offline_no_led)
workflow.add_node("check_LED_yes", yes_led_yes)
workflow.add_node("check_PDU", yes_led_no)
workflow.add_node("check_PDU_yes", same_pdu_yes)
workflow.add_node("check_PDU_no", same_pdu_no)

# Add edges

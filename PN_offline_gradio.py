from langgraph.graph import Graph
import gradio as gr


# Node Definitions
def input_status_node(state):
    return {
        "current_status": "Device Offline",
        "original_status": "Device Offline",
        "messages": ["Initial status: Device Offline"]
    }


def offline_operation(state):
    return {
        "current_status": 'PWD/PSU check',
        "original_status": state['original_status'],
        "messages": state["messages"] + [
            "\nAre all the Power cables seated at both ends? And are the PSUs completely seated and present?"],
        "question": "pwd_check"
    }


def process_pwd_response(state, response):
    if response == "YES":
        return {"path": "YES", **state}
    else:
        return {"path": "NO", **state}


def offline_yes_led(state):
    return {
        "current_status": 'LED check',
        "original_status": state['original_status'],
        "messages": state["messages"] + ["\nAre the front/back LED's showing any light?"],
        "question": "led_check"
    }


def process_led_response(state, response):
    if response == "YES":
        return {"path": "YES", **state}
    else:
        return {"path": "NO", **state}


def offline_no_led(state):
    return {
        "current_status": "Reverted to Offline operation",
        "original_status": state['original_status'],
        "messages": state["messages"] + ["\nAfter ensuring all PWD/PSU connections, device should power ON"],
        "path": ''
    }


def yes_led_yes(state):
    return {
        "current_status": 'Final recommendation',
        "original_status": state['original_status'],
        "messages": state["messages"] + [
            "\nAsk Engineer to console device and troubleshoot. Device or component must be replaced"],
        "path": ''
    }


def yes_led_no(state):
    return {
        "current_status": 'PDU check',
        "original_status": state['original_status'],
        "messages": state["messages"] + ["\nDo any other devices connected to same PDU have power?"],
        "question": "pdu_check"
    }


def process_pdu_response(state, response):
    if response == "YES":
        return {"path": "YES", **state}
    else:
        return {"path": "NO", **state}


def same_pdu_yes(state):
    return {
        "current_status": 'Final recommendation',
        "original_status": state['original_status'],
        "messages": state["messages"] + [
            "\nInform Engineering: 'Other devices connected have PWR' - PDU could be faulty. Switch Router!"],
        "path": ''
    }


def same_pdu_no(state):
    return {
        "current_status": 'Final recommendation',
        "original_status": state['original_status'],
        "messages": state["messages"] + [
            "\nInform Engineering: 'Other devices connected do not have PWR' - PDU isn't powered."],
        "path": ''
    }


# Create and configure graph
workflow = Graph()

# Add nodes
workflow.add_node("input_offline", input_status_node)
workflow.add_node("check_PWD_cables", offline_operation)
workflow.add_node("process_pwd", process_pwd_response)
workflow.add_node("check_LED", offline_yes_led)
workflow.add_node("process_led", process_led_response)
workflow.add_node("offline_PWD_no", offline_no_led)
workflow.add_node("check_LED_yes", yes_led_yes)
workflow.add_node("check_PDU", yes_led_no)
workflow.add_node("process_pdu", process_pdu_response)
workflow.add_node("check_PDU_yes", same_pdu_yes)
workflow.add_node("check_PDU_no", same_pdu_no)

# Add edges
workflow.add_edge("input_offline", "check_PWD_cables")
workflow.add_conditional_edges("process_pwd",lambda state: state['path'],{'YES': "check_LED", 'NO': "offline_PWD_no"})
workflow.add_edge("offline_PWD_no", "check_PWD_cables")
workflow.add_conditional_edges(
    "process_led",
    lambda state: state['path'],
    {'YES': 'check_LED_yes', 'NO': 'check_PDU'}
)
workflow.add_conditional_edges(
    "process_pdu",
    lambda state: state['path'],
    {'YES': 'check_PDU_yes', 'NO': 'check_PDU_no'}
)

# Set entry and finish points
workflow.set_entry_point("input_offline")
workflow.set_finish_point("check_LED_yes")
workflow.set_finish_point("check_PDU_yes")
workflow.set_finish_point("check_PDU_no")

# Compile the workflow
app = workflow.compile()


# Gradio Interface
class WorkflowRunner:
    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.current_state = {"input": "Device Offline"}
        self.next_node = "input_offline"
        self.current_question = None

    def run_step(self, response=None):
        # Process response if provided
        if response and self.current_question:
            if self.current_question == "pwd_check":
                self.current_state = process_pwd_response(self.current_state, response)
                self.next_node = "check_LED" if response == "YES" else "offline_PWD_no"
            elif self.current_question == "led_check":
                self.current_state = process_led_response(self.current_state, response)
                self.next_node = "check_LED_yes" if response == "YES" else "check_PDU"
            elif self.current_question == "pdu_check":
                self.current_state = process_pdu_response(self.current_state, response)
                self.next_node = "check_PDU_yes" if response == "YES" else "check_PDU_no"
            self.current_question = None

        # Run through nodes until we hit a question or finish
        while self.next_node:
            # Execute current node
            result = app.nodes[self.next_node].invoke(self.current_state)
            self.current_state.update(result)

            # Check if finished
            if self.next_node in ["check_LED_yes", "check_PDU_yes", "check_PDU_no"]:
                return "\n".join(self.current_state["messages"]), None, None, None

            # Check if we have a question to ask
            if "question" in result:
                self.current_question = result["question"]
                visibility = {
                    "pwd_check": (True, False, False),
                    "led_check": (False, True, False),
                    "pdu_check": (False, False, True)
                }[self.current_question]
                return (
                    "\n".join(self.current_state["messages"]),
                    *visibility
                )

            # Get next node from edges
            if self.next_node == "input_offline":
                self.next_node = "check_PWD_cables"
            elif self.next_node == "check_PWD_cables":
                self.next_node = "process_pwd"
            elif self.next_node == "offline_PWD_no":
                self.next_node = "check_PWD_cables"
            elif self.next_node == "check_LED":
                self.next_node = "process_led"
            elif self.next_node == "check_PDU":
                self.next_node = "process_pdu"
            elif self.next_node in ["process_pwd", "process_led", "process_pdu"]:
                # These nodes are handled by the conditional edges
                self.next_node = None
                continue
            else:
                self.next_node = None

        return "\n".join(self.current_state["messages"]), None, None, None


workflow_runner = WorkflowRunner()


def handle_response(response):
    output, pwd_vis, led_vis, pdu_vis = workflow_runner.run_step(response)
    return output, gr.update(visible=pwd_vis), gr.update(visible=led_vis), gr.update(visible=pdu_vis)


def reset_conversation():
    workflow_runner.reset_state()
    output, pwd_vis, led_vis, pdu_vis = workflow_runner.run_step()
    return output, gr.update(visible=pwd_vis), gr.update(visible=led_vis), gr.update(visible=pdu_vis)


with gr.Blocks() as demo:
    gr.Markdown("# Network Device Troubleshooting Workflow")
    gr.Markdown("### Starting with device status: Offline")

    with gr.Row():
        output = gr.Textbox(label="Troubleshooting Steps", lines=10, interactive=False)

    with gr.Row():
        with gr.Column():
            pwd_check = gr.Radio(choices=["YES", "NO"], label="Power Check Response", visible=False)
        with gr.Column():
            led_check = gr.Radio(choices=["YES", "NO"], label="LED Check Response", visible=False)
        with gr.Column():
            pdu_check = gr.Radio(choices=["YES", "NO"], label="PDU Check Response", visible=False)

    with gr.Row():
        reset_btn = gr.Button("Restart Troubleshooting")

    # Initial run
    initial_output, pwd_vis, led_vis, pdu_vis = workflow_runner.run_step()
    output.value = initial_output

    # Event handlers
    pwd_check.select(
        fn=handle_response,
        inputs=pwd_check,
        outputs=[output, pwd_check, led_check, pdu_check]
    )

    led_check.select(
        fn=handle_response,
        inputs=led_check,
        outputs=[output, pwd_check, led_check, pdu_check]
    )

    pdu_check.select(
        fn=handle_response,
        inputs=pdu_check,
        outputs=[output, pwd_check, led_check, pdu_check]
    )

    reset_btn.click(
        fn=reset_conversation,
        outputs=[output, pwd_check, led_check, pdu_check]
    )

if __name__ == "__main__":
    demo.launch()
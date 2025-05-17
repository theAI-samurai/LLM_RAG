from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_core.messages import HumanMessage
import json
from environment import AUTH_TYPE, COMPARTMENT_ID, GENAI_ENDPOINT, MODEL_ID, MODEL_ID_CR, OCI_PROFILE, REGION

COMPARTMENT_ID= "ocid1.compartment.oc1..aaaaaaaay2vcslh7hynw4fqfs476lsqyvgptfyvzfdbcqewqpc4qhmim5nqa"
MODEL_ID_CR = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq"
OCI_PROFILE = "ocidataaienablement"

class OCILLM:
    def __init__(self, model_name=MODEL_ID):
        """
            Initialize the object with the model attribute set to "cohere.command-r".
        """
        self.model_name = model_name
        self.config_profile = OCI_PROFILE
        self.llm = ChatOCIGenAI(
            model_id=self.model_name,
            service_endpoint=GENAI_ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            auth_profile=OCI_PROFILE,
            auth_type=AUTH_TYPE,
            model_kwargs={
                "temperature": 1,
                "max_tokens": 600,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "top_k": 0,
                "top_p": 0.75
                          },
            provider="cohere",
        )

    def generate_answer(self, documents, preamble=None, prompt=None, to_lang=None):
        """
            Generate the chat response using the provided preamble, prompt, and documents.

            Parameters:
                preamble (str): The text to set as the preamble override.
                prompt (str): The text prompt for the chat response.
                documents (list): A list of documents to consider during chat generation.

            Returns:
                str: The generated chat response text.
        """

        if AUTH_TYPE == "API_KEY":

            if to_lang is not None:
                # translator is Running
                languages = {"en": "English", "ja": "Japanese"}
                preamble = "You are a helpful AI assistant to help translate text from one language to another"
                prompt = f"Please convert the follow text into {languages[to_lang]} Please maintain same paragraphs structure."

                try:
                    translated_docs = [self.generate_answer(documents=[doc], preamble=preamble, prompt=prompt,to_lang=None) for doc in documents]
                    return translated_docs
                except Exception as e:
                    print(e)
                return

            else:
                config = {"region": REGION}
                # Combine preamble, documents, and prompt into a structured message
                full_context = f"""{preamble}
                                    DOCUMENTS:{documents}
                                    QUESTION:{prompt}
                                    Please answer the question based on the provided documents and context.
                                    """
                messages = [HumanMessage(content=full_context), ]
                chat_response = self.llm.invoke(messages)
                chat_response_vars = vars(chat_response)
                try:
                    resp_json = chat_response_vars["data"]                # json.loads(str(chat_response_vars["data"]))
                except:
                    resp_json = str(chat_response_vars["content"])
                res = resp_json     # ["chat_response"]["text"]

                return res



# obj = OCILLM(MODEL_ID_CR)
# res = obj.generate_answer(documents=['This is sample sentence', 'Hi! I am Ankit Mishra, Good Morning!'], to_lang='ja')
#




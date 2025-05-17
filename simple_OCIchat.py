from environment import (AUTH_TYPE, COMPARTMENT_ID, GENAI_ENDPOINT, MODEL_ID, MODEL_ID_CR, OCI_PROFILE, REGION,)
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.messages import HumanMessage


COMPARTMENT_ID= "ocid1.compartment.oc1..aaaaaaaay2vcslh7hynw4fqfs476lsqyvgptfyvzfdbcqewqpc4qhmim5nqa"
MODEL_ID_CR = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq"
CONFIG_PROFILE = "ocidataaienablement"
# endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
prompt="translate 'HOLA' to english"  # {input}

# initialize interface
chat = ChatOCIGenAI(
  model_id=MODEL_ID_CR,
  service_endpoint=GENAI_ENDPOINT,
  compartment_id=COMPARTMENT_ID,
  provider="cohere",
  model_kwargs={
    "temperature": 1,
    "max_tokens": 600,
    "frequency_penalty": 0,
    "presence_penalty": 0,
	"top_k": 0,
    "top_p": 0.75
  },
  auth_type=AUTH_TYPE,
  auth_profile=CONFIG_PROFILE
)

messages = [HumanMessage(content=prompt),]

response = chat.invoke(messages)

# Print result
print("**************************Chat Result**************************")
print(vars(response))
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from llm import *
from langchain.memory import ConversationBufferMemory


OCI_EMBED_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

# Load and split the document
loader = PyPDFLoader("basics-of-data-science-kpk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(pages)

# ----------    CREATING EMBEDDING MODEL FROM ORACLE    --------------
embeddings = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=OCI_EMBED_ENDPOINT,
    truncate="NONE",
    compartment_id=COMPARTMENT_ID,
    auth_type=AUTH_TYPE,
    auth_profile=OCI_PROFILE
)

# ----------   VECTOR STORE + EMBEDDING     ------ Choma DB ----------
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# -----------   OCI LLM (cohere) + LLM invoke  -----------------
llm = OCILLM(model_name=MODEL_ID_CR)


# ---------------------------- Initialize Conversation Memory ----------
memory = ConversationBufferMemory(memory_key="chat_history",        # stores past interactions
                                  return_message=True               # Returns message in list format
                                  )

#       ----------------------  METHOD 2    ---------------------------
#  ---------                 Langchain CHAT LOOP    ------------------- (better use this)
while True:
    preamble = "You are a question and answer assistant."
    query = input("YOU : ")
    if query.lower().__contains__("exit") or query.lower().__contains__("quit"):
        break
    relevant_docs = retriever.get_relevant_documents(query)
    chat_history = memory.load_memory_variables({})['chat_history']         # get conv history from memory
    full_prompt = f"""Previous conversation :{chat_history} \n New Question : {query}\n"""
    res = llm.generate_answer(documents=relevant_docs, preamble=preamble, prompt=full_prompt)
    print(res)


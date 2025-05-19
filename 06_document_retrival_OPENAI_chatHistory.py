
"""
Name        :       Ankit Mishra
Description :       This code replicates langchain based chat model with OPENAI
                    Function openai_generate_answer is created like OCI llm in OCI examples.

NOTE        :       Langchain has most modules in  langchain_community now.

                    For OpenAI most functions are in langchain_openai.
"""

from environment import OPENAI_ANKIT
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# openai based llm calls
from langchain.schema import HumanMessage


def openai_generate_answer(documents, preamble, prompt, llm):
    """
        Generate the chat response using the provided preamble, prompt, and documents.

        Parameters:
            preamble (str): The text to set as the preamble override.
            prompt (str): The text prompt for the chat response.
            documents (list): A list of documents to consider during chat generation.

        Returns:
            str: The generated chat response text.
    """
    full_context = f"""{preamble}
                    DOCUMENTS:{documents}
                    QUESTION:{prompt}
                    Please answer the question based on the provided documents and context.
                    """
    messages = [HumanMessage(content=full_context)]
    chat_response = llm.invoke(messages)
    chat_response_vars = vars(chat_response)
    try:
        resp_json = chat_response_vars["data"]  # json.loads(str(chat_response_vars["data"]))
    except:
        resp_json = str(chat_response_vars["content"])
    res = resp_json  # ["chat_response"]["text"]

    return res


loader = PyPDFLoader("basics-of-data-science-kpk.pdf")
pages = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(pages)

# --------------EMBEDDING MODEL  from   OPENAI  -------------------
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",
                              openai_api_key = OPENAI_ANKIT
                              )

# ----------   VECTOR STORE + EMBEDDING     ------ Choma DB ----------
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------ OPEN AI LLM    ------------------------------
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_ANKIT)

#   ----------------    Initialize Conversational Memory    ---------
memory = ConversationBufferMemory(memory="history",
                                  return_message=True)

#  ------------------     Langchain CHAT LOOP    -------------------
while True:
    preamble = "You are a question and answer assistant."
    query = input("YOU : ")
    if query.lower().__contains__("exit") or query.lower().__contains__("quit"):
        break
    relevant_docs = retriever.get_relevant_documents(query)
    chat_history = memory.load_memory_variables({})['history']         # get conv history from memory
    full_prompt = f"""Previous conversation :{chat_history} \n New Question : {query}\n"""

    res = openai_generate_answer(documents=relevant_docs,
                                 preamble=preamble,
                                 prompt=full_prompt,
                                 llm=llm
                                 )
    print(res)


from dotenv import load_dotenv
load_dotenv()
import os
import pymupdf  # Alternative import for PyMuPDF
import requests
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent, load_tools

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 4096,
    "min_new_tokens": 1,
    "temperature": 0,
    "repetition_penalty": 1
}

Wx_Api_Key = os.getenv("WX_API_KEY", None)
Project_ID = os.getenv("PROJECT_ID", None)
cloud_url = os.getenv("IBM_CLOUD_URL", None)

vector_stores = {}

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def get_documents_from_pdf(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()

    # Open the PDF file
    pdf_document = pymupdf.open(stream=response.content, filetype="pdf")
    text = ""

    # Extract text from each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    # Close the PDF file
    pdf_document.close()

    # Split the extracted text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    splitDocs = splitter.create_documents([text])
    return splitDocs

def web_search_result(text):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    value = wikipedia.run(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    splitDocs = splitter.split_documents(value)
    return splitDocs

def create_db(docs):
    embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
        url=cloud_url,
        project_id=Project_ID,
        params=parameters,
        apikey=Wx_Api_Key,
    )
    vectorStore = FAISS.from_documents(docs, embedding=embedding)
    return vectorStore

def create_chain(vectorStore):
    model = WatsonxLLM(
        model_id="meta-llama/llama-3-70b-instruct",
        url=cloud_url,
        project_id=Project_ID,
        params=parameters,
        apikey=Wx_Api_Key,
        verbose=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the context: {context} "),
        MessagesPlaceholder(variable_name='chat_history'),
        ("human", "{input}") 
    ])
    
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vectorStore.as_retriever(search_kwargs={"k":15})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human", "Given the above conversation, generate a search query to look up in order to get information relavant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    retriever_chain = create_retrieval_chain(
        history_aware_retriever,
        chain
    )

    return retriever_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    return response["answer"]

def get_user_chain(uid):
    if uid not in vector_stores:
        # Load documents for the user, for now we use a static URL for the example
        pdf_url = 'https://utfs.io/f/27ca8fdc-1745-4f4c-b764-512c01392a29-uijmwi.pdf'
        docs = get_documents_from_pdf(pdf_url)
        vectorStore = create_db(docs)
        vector_stores[uid] = vectorStore
    else:
        vectorStore = vector_stores[uid]
    
    return create_chain(vectorStore)

if __name__ == '__main__':
    uid = input("Enter user ID: ")
    chain = get_user_chain(uid)
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Chat history content is now:  ", chat_history)
        print("Assistant: ", response)

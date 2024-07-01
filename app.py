from dotenv import load_dotenv
load_dotenv()
import os
import pymupdf  # Alternative import for PyMuPDF
import requests
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
# from langchain_ibm import ChatWatsonx
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from google.generativeai import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.agent_toolkits.load_tools import load_tools
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


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 400,
        chunk_overlap = 20
    )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

# def get_citation():
#     llm = ChatWatsonx(
#         model_id="ibm/granite-13b-chat-v2",
#         url=cloud_url,
#         project_id=Project_ID,
#         params=parameters,
#     )
#     # llm = WatsonxLLM(
#     #         model_id="meta-llama/llama-3-8b-instruct",
#     #         # model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q",
#     #         url=cloud_url,
#     #         project_id=Project_ID,
#     #         params=parameters,
#     #         apikey=Wx_Api_Key,
#     #         verbose=True
#     #     )
#     tools = load_tools(
#         ["arxiv"],
#     )
#     prompt = hub.pull("hwchase17/react")

#     agent = create_react_agent(llm, tools, prompt)
#     agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#     agent_executor.invoke(
#         {
#             "input": "What's the paper about Tamil LLaMA?",
#         }
#     )


def get_documents_from_pdf(pdf_path):
    response = requests.get(pdf_path)
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
        chunk_size = 400,
        chunk_overlap = 20
    )
    splitDocs = splitter.create_documents([text])
    print(splitDocs)
    return splitDocs

def web_search_result(text):
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    value = wikipedia.run(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size = 400,
        chunk_overlap = 20
    )
    splitDocs = splitter.split_documents(value)
    return splitDocs
    
# def video_transcript_creator():


# def create_db(docs):
#     embedding = OpenAIEmbeddings()
#     vectorStore = FAISS.from_documents(docs, embedding=embedding)
#     return vectorStore

def create_db(docs):
    # embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    embedding = WatsonxEmbeddings(
        model_id="ibm/slate-125m-english-rtrvr",
            # model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q",
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
            #model_id="ibm-mistralai/mixtral-8x7b-instruct-v01-q",
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
        #retriever,
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

# if __name__ == '__main__':
    # web_search_result()
    # get_citation()

if __name__ == '__main__':
    # docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
    pdf_path = 'https://utfs.io/f/27ca8fdc-1745-4f4c-b764-512c01392a29-uijmwi.pdf'
    
    docs = get_documents_from_pdf(pdf_path)


    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)

    # chat_history = [
    #     HumanMessage(content='Hello'),
    #     AIMessage(content="Hello, how can I assist you? "),
    #     HumanMessage(content="My name is Girijesh")
    # ]
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
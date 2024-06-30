import requests as rq
from dotenv import load_dotenv
load_dotenv()
import os
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
# from google.generativeai import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic

API_KEY = ''

app = Flask(__name__)
CORS(app)  # Enable CORS
client = anthropic.Client(api_key=API_KEY)
app.config['SECRET_KEY'] = 'secret!'


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

# def create_db(docs):
#     embedding = OpenAIEmbeddings()
#     vectorStore = FAISS.from_documents(docs, embedding=embedding)
#     return vectorStore

def create_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
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

    retriever = vectorStore.as_retriever(search_kwargs={"k":3})

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


@app.post("/init")
def init_fucntion():
    docs = get_documents_from_web('https://python.langchain.com/docs/expression_language/')
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

@app.post("/store")
def msg():
    chat_t = request.json.get("chat")
    uid = chat_t["userid"]
    chat_h = request.json.get("chat_history")

    # Ensure directories exist
    os.makedirs("outputs", exist_ok=True)  # Create 'outputs' if it doesn't exist
    os.makedirs(f"outputs/{uid}", exist_ok=True)  # Create user-specific directory

    chat_p = f"outputs/{uid}/chat.json"
    chath_p = f"outputs/{uid}/chat_history.json"
    st = 1
    try:
        # Load existing data or initialize empty lists
        try:
            with open(chat_p, "r") as f:
                file_content = f.read().strip()  # Remove potential leading/trailing whitespace
                if file_content:  # Check if content is not empty
                    existing_chat_data = json.loads(file_content)
                else:
                    if chat_t in existing_chat_data:
                        existing_chat_data = []
                        st = 0
        except FileNotFoundError:
            existing_chat_data = []

        try:
            with open(chath_p, "r") as f:
                file_content = f.read().strip()  # Remove potential leading/trailing whitespace
                if file_content:  # Check if content is not empty
                    existing_chat_history_data = json.loads(file_content)
                else:
                    existing_chat_history_data = []
        except FileNotFoundError:
            existing_chat_history_data = []

        # Append new chat data
        existing_chat_data.append(chat_t)

        # Append new chat history data
        existing_chat_history_data.append(chat_h)

        # Write updated data back to files
        if st:
            with open(chat_p, "w") as f:
                json.dump(existing_chat_data, f, indent=2)

        with open(chath_p, "w") as f:
            json.dump(existing_chat_history_data, f, indent=2)

        return jsonify({"Success": "Database store ok"}), 200

    except json.JSONDecodeError as json_error:
        print(f"Error in msg: Invalid JSON in file - {json_error}")  # Provide detailed error
        return jsonify({"error": f"Invalid JSON in file: {json_error}"}), 400  # Return 400 for bad request

    except Exception as error:
        print(f"Error in msg: {error}")  # General error handling
        return jsonify({"error": str(error)}), 500 
    

@app.post("/filestr")
def file():
    file = request.json.get("file")
    uid = file["file"]["uid"]
    os.makedirs("outputs", exist_ok=True)  # Create 'outputs' if it doesn't exist
    os.makedirs(f"outputs/{uid}", exist_ok=True)  # Create user-specific directory

    file_p = f"outputs/{uid}/files.json"
    with open(file_p, "w+") as f:
        print("file created")


    try:
        # Load existing data or initialize empty lists
        try:
            with open(file_p, "r+") as f:
                file_content = f.read().strip()  # Remove potential leading/trailing whitespace
                if file_content:  # Check if content is not empty
                    existing_files = json.loads(file_content)
                else:
                    existing_files = []
                    
        except FileNotFoundError:
            existing_files = []


        # Append new chat data
        existing_files.append(file_p)

        # Write updated data back to files

        with open(file_p, "w+") as f:
            json.dump(existing_files, f, indent=2)

        return jsonify({"Success": "Database store ok"}), 200

    except json.JSONDecodeError as json_error:
        print(f"Error in msg: Invalid JSON in file - {json_error}")  # Provide detailed error
        return jsonify({"error": f"Invalid JSON in file: {json_error}"}), 400  # Return 400 for bad request

    except Exception as error:
        print(f"Error in msg: {error}")  # General error handling
        return jsonify({"error": str(error)}), 500 


    
@app.get("/")
def home():
    return "Welcome to the Rune-Backend AI API"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
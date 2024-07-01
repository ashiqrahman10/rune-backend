from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import pickle
import pymupdf  # Alternative import for PyMuPDF
import requests
import sqlite3
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

db_conn = sqlite3.connect('vector_stores.db', check_same_thread=False)
cursor = db_conn.cursor()

# Create tables for storing vector stores and chat history
cursor.execute('''
CREATE TABLE IF NOT EXISTS vector_stores (
    uid TEXT PRIMARY KEY,
    documents BLOB
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    uid TEXT,
    role TEXT,
    message TEXT
)
''')

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

def save_vector_store(uid, docs):
    docs_blob = pickle.dumps(docs)
    cursor.execute('REPLACE INTO vector_stores (uid, documents) VALUES (?, ?)', (uid, docs_blob))
    db_conn.commit()

def load_vector_store(uid):
    cursor.execute('SELECT documents FROM vector_stores WHERE uid=?', (uid,))
    row = cursor.fetchone()
    if row:
        docs_blob = row[0]
        docs = pickle.loads(docs_blob)
        return docs
    return None

def create_chain_from_docs(docs):
    vectorStore = create_db(docs)
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

def save_chat_history(uid, chat_history):
    cursor.execute('DELETE FROM chat_history WHERE uid=?', (uid,))
    for message in chat_history:
        role = 'human' if isinstance(message, HumanMessage) else 'ai'
        cursor.execute('INSERT INTO chat_history (uid, role, message) VALUES (?, ?, ?)', (uid, role, message.content))
    db_conn.commit()

def load_chat_history(uid):
    cursor.execute('SELECT role, message FROM chat_history WHERE uid=?', (uid,))
    rows = cursor.fetchall()
    chat_history = []
    for role, message in rows:
        if role == 'human':
            chat_history.append(HumanMessage(content=message))
        else:
            chat_history.append(AIMessage(content=message))
    return chat_history

def get_user_chain(uid, docs=None):
    if docs:
        save_vector_store(uid, docs)
    else:
        docs = load_vector_store(uid)
    if not docs:
        docs = [""]

    return create_chain_from_docs(docs)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    uid = data.get('uid')
    user_input = data.get('message')
    pdf_url = data.get('pdf_url', None)
    
    if not uid or not user_input:
        return jsonify({"error": "Invalid input"}), 400

    if pdf_url:
        docs = get_documents_from_pdf(pdf_url)
        chain = get_user_chain(uid, docs)
    else:
        chain = get_user_chain(uid)
        

    if not chain:
        return jsonify({"error": "No document found for user"}), 400
    
    chat_history = load_chat_history(uid)
    
    response = process_chat(chain, user_input, chat_history)
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    save_chat_history(uid, chat_history)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

db_conn.close()

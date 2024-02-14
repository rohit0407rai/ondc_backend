from flask import Flask, jsonify, request
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from marshmallow import Schema, fields, validate, ValidationError
import mimetypes
from werkzeug.utils import secure_filename

import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import requests
from io import BytesIO
from langchain.memory import ConversationBufferMemory


app = Flask(__name__)
CORS(app)

load_dotenv()  # load all our environment variables

genai.configure(api_key="AIzaSyAUV0qh7tfsK1cwWWr3c9fxk6NS17UWPyE")

def get_gemini_response(prompt):
    fixed_prompt = """
    {
        "Checklist": [
            answer,
            answer,
            answer,
            answer,
            answer,
            answer,
            answer,
            answer,
            answer,
            answer
        ]
    }
"""
# Move input_prompt definition inside the function
    input_prompt = f"""
    You are a compliance expert and I need you to generate a compliance list in the following format:
    {fixed_prompt}
    keep the key values as it is, instead of answer put the thing to be done for compliance
    These final answer should in array format as menioned above as mentioned before, you need to do the following making sure all of the above is followed:{prompt}
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    return response.text


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.get_json()
    prompt = data['prompt']

    response_text = get_gemini_response(prompt)

    # Remove triple backticks from the beginning and end of the response
    if response_text.startswith("```") and response_text.endswith("```"):
        response_text = response_text[3:-3]

    # Check if the response_text starts with "json" and remove it
    if response_text.startswith("json"):
        response_text = response_text[4:]

    return jsonify({'response': response_text})





# Initialize Firebase
cred = credentials.Certificate("backend/user_data/ondcproject-b8d10-firebase-adminsdk-kt8cw-457fd65bc5.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'userData'})

db = firestore.client()
bucket = storage.bucket()

def upload_file(file_path):
    # Upload file to Firebase Storage
    blob = bucket.blob("userData")
    blob.upload_from_filename(file_path)

    # Get download URL
    blob_url = blob.generate_signed_url()  # No expiration time set
    return blob_url


@app.route('/add_data', methods=['POST'])
def add_data():

    userData = {
    "first_name": request.json.get("first_name"),
    "middle_name": request.json.get("middle_name"),
    "last_name": request.json.get("last_name"),
    "email": request.json.get("email"),
    "alternate_email": request.json.get("alternate_email"),
    "date_of_birth": request.json.get("date_of_birth"),
    "gender": request.json.get("gender"),
    "Phone_number": request.json.get("Phone_number"),
    "occupation": request.json.get("occupation"),
    "Nationality": request.json.get("Nationality"),
    "marital_status": request.json.get("marital_status"),
    "education": request.json.get("education"),
    "income": request.json.get("income"),
    "purpose_of_account": request.json.get("purpose_of_account"),
    "benificiary_details": request.json.get("benificiary_details"),
    "tax_residency_status": request.json.get("tax_residency_status"),
    "politically_exposed_person_status": request.json.get("politically_exposed_person_status"),
    }



    # Omit the argument to add() to let Firestore generate an ID
    doc_ref = db.collection('userData').add(userData)

    # Extract the automatically generated document ID
    document_id = doc_ref[1].id if len(doc_ref) > 1 else None

    return jsonify({"message": "Data added successfully", "document_id": document_id})

@app.route('/add_docs', methods=['POST'])
def add_docs():

    proof_of_identity = request.files.get("proof_of_identity")
    proof_of_incorporation = request.files.get("proof_of_incorporation")
    proof_of_address = request.files.get("proof_of_address")
    proof_of_company_address = request.files.get("proof_of_company_address")
    proof_of_income = request.files.get("proof_of_income")
    pan_number_of_company = request.files.get("pan_number_of_company")
    photo = request.files.get("photo")
    gst_reg = request.files.get("gst_reg")

    user_docs = {
    "proof_of_identity":  upload_file(proof_of_identity),
    "proof_of_incorporation": upload_file(proof_of_incorporation),
    "proof_of_address": upload_file(proof_of_address),
    "proof_of_company_address": upload_file(proof_of_company_address),
    "proof_of_income": upload_file(proof_of_income),
    "pan_number_of_company":upload_file(pan_number_of_company),
    "photo": upload_file(photo),
    "gst_reg": upload_file(gst_reg)
    }

    doc_ref = db.collection('userDocs').add(user_docs)

    # Extract the automatically generated document ID
    document_id = doc_ref[1].id if len(doc_ref) > 1 else None

    return jsonify({"message": "Data added successfully", "document_id": document_id})

# @app.route('/add_docs', methods=['POST'])
# def add_docs():
#     # Ensure all required files are present in the request
#     required_files = [
#         "proof_of_identity",
#         "proof_of_incorporation",
#         "proof_of_address",
#         "proof_of_company_address",
#         "proof_of_income",
#         "pan_number_of_company",
#         "photo",
#         "gst_reg"
#     ]
#     for file_key in required_files:
#         if file_key not in request.files:
#             return jsonify({"error": f"File '{file_key}' is missing"}), 400

#     # Upload files to Firebase Storage
#     user_docs = {}
#     for file_key in required_files:
#         file = request.files[file_key]
#         if file:
#             file_url = upload_file(file)
#             user_docs[file_key] = file_url
#         else:
#             return jsonify({"error": f"Failed to upload '{file_key}' file"}), 500

#     # Add the uploaded document URLs to Firestore
#     doc_ref = db.collection('userDocs').add(user_docs)

#     # Extract the automatically generated document ID
#     document_id = doc_ref.id

#     return jsonify({"message": "Data added successfully", "document_id": document_id})


@app.route('/additional_data', methods=['POST'])
def additional_data():
    userData = {
            "purpose_of_account": request.json.get("purpose_of_account"),
            "benificiary_details": request.json.get("benificiary_details"),
            "tax_residency_status": request.json.get("tax_residency_status"),
            "politically_exposed_person_status": request.json.get("politically_exposed_person_status"),
            "declaration_of_financial_statment": request.json.get("declaration_of_financial_statment"),
            "signature": request.json.get("signature"),     
         }
    
    doc_ref = db.collection('additional_data').add(userData)

    document_id = doc_ref[1].id if len(doc_ref) > 1 else None

    return jsonify({"message": "Data added successfully", "document_id": document_id})

        
         
         
@app.route('/get_data', methods=['GET'])
def get_collection_data():
    collection_data = []

    # Fetch data from the desired collection
    collection_ref = db.collection('agriculture').get()
    for doc in collection_ref:
        collection_data.append(doc.to_dict())

    # Return the data from the collection
    return jsonify(collection_data)



pdf_url = None
conversation = None

def get_pdf_text(pdf_url):
    # Fetch the PDF from the URL
    response = requests.get(pdf_url)
    if response.status_code != 200:
        raise Exception("Failed to fetch PDF from URL")

    # Read PDF content
    pdf_content = BytesIO(response.content)

    # Extract text from the PDF
    pdf_reader = PdfReader(pdf_content)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def save_pdf_url(pdf_url):
    with open('pdf_url.txt', 'w') as f:
        f.write(pdf_url)

def load_pdf_url():
    with open('pdf_url.txt', 'r') as f:
        return f.read().strip()


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory  # You can add memory if needed
    )
    return conversation_chain



@app.route('/upload_pdf_url', methods=['POST'])
def upload_pdf_url():
    global pdf_url, conversation
    pdf_url = request.json.get('pdf_url')
    save_pdf_url(pdf_url)
    print(pdf_url)
    pdf_url = load_pdf_url()
    if pdf_url is None:
        return jsonify({'error': 'PDF URL not provided'}), 400

    raw_text = get_pdf_text(pdf_url)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    return jsonify({'message': 'Conversation chain started successfully'})


# @app.route('/start_conversation', methods=['GET'])
# def start_conversation():
#     global pdf_url, conversation
#     pdf_url = load_pdf_url()
#     if pdf_url is None:
#         return jsonify({'error': 'PDF URL not provided'}), 400

#     raw_text = get_pdf_text(pdf_url)
#     text_chunks = get_text_chunks(raw_text)
#     vectorstore = get_vectorstore(text_chunks)
#     conversation = get_conversation_chain(vectorstore)

#     return jsonify({'message': 'Conversation chain started successfully'})


@app.route('/chat', methods=['POST'])
def chat():
    global conversation
    if conversation is None:
        return jsonify({'error': 'Conversation chain not initialized'}), 400

    user_question = request.json.get('question')
    response = conversation({'question': user_question})

    # Extract content from AIMessage if present
    print(response['answer'])


    return jsonify({'response': response['answer']})


    
   



# @app.route('/get_data', methods=['GET'])
# def get_data():
#     data = []
#     docs = db.collection('userData').stream()

#     for doc in docs:
#         data.append(doc.to_dict())

#     return jsonify(data)

# @app.route('/get_docs', methods=['GET'])
# def get_data():
#     data = []
#     docs = db.collection('userDocs').stream()

#     for doc in docs:
#         data.append(doc.to_dict())

#     return jsonify(data)


# @app.route('/get_additional_data', methods=['GET'])
# def get_data():
#     data = []
#     docs = db.collection('additional_data').stream()

#     for doc in docs:
#         data.append(doc.to_dict())

#     return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

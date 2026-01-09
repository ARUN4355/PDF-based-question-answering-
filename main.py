import os
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt
from werkzeug.utils import secure_filename

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import easyocr
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
from transformers import pipeline

UPLOAD_FOLDER = "uploads"
DB_FOLDER = "faiss_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "secret123"
jwt = JWTManager(app)
# JWT blacklist
blacklisted_tokens = set()
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    return jti in blacklisted_tokens

# ---------------- AUTH ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if data["username"] == "admin" and data["password"] == "admin@9":
        return jsonify(access_token=create_access_token(identity="admin"))
    return jsonify(msg="Bad credentials"), 401


# ---------------- OCR ----------------
reader = easyocr.Reader(['en'])

def ocr_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "This PDF is scanned and requires Poppler. "
            "Please install Poppler from https://github.com/oschwartz10612/poppler-windows"
        )

    text = ""
    for img in images:
        result = reader.readtext(img)
        for (_, t, _) in result:
            text += t + " "
    return text 


# ---------------- PDF TEXT ----------------
def extract_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = " ".join([p.page_content for p in pages])

    if sum(c.isalnum() for c in full_text) < 50:
        print("Using OCR...")
        full_text = ocr_pdf(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([full_text])


# ---------------- VECTOR DB ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_db(docs):
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FOLDER)

def load_db():
    return FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)


# ---------------- OFFLINE LLM ----------------
llm = pipeline("text2text-generation", model="google/flan-t5-small")

def ask_llm(context, query):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    return llm(prompt, max_new_tokens=200)[0]["generated_text"]


# ---------------- FILE UPLOAD ----------------
@app.route("/fileupload", methods=["POST"])
@jwt_required()
def upload():
    file = request.files["file"]
    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    try:
        docs = extract_text(path)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    build_db(docs)

    return jsonify({"message": "PDF processed successfully"})


# ---------------- QUERY ----------------
@app.route("/query", methods=["POST"])
@jwt_required()
def query():
    q = request.json["query"]
    db = load_db()
    docs = db.similarity_search(q, k=4)

    context = "\n".join([d.page_content for d in docs])
    answer = ask_llm(context, q)

    return jsonify({"answer": answer})


# -----------------LOGOUT------------------ 
@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    jti = get_jwt()["jti"]
    blacklisted_tokens.add(jti)
    return jsonify({"message": "Logged out successfully"})


if __name__ == "__main__":
    app.run()

#{"username": "admin", "password": "admin@9"}


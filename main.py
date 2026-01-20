import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- BASIC CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

USERS = {"admin": "admin@9"}
USER_INDEX = {}

# ---------------- FLASK APP ----------------
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "nalco-secret"
jwt = JWTManager(app)

# ---------------- EMBEDDINGS (FAST, CPU-SAFE) ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    valid_pages = []
    total_chars = 0

    for p in pages:
        text = p.page_content.strip()
        if len(text) > 100:
            valid_pages.append(p)
            total_chars += len(text)

    if not valid_pages or total_chars < 1000:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_documents(valid_pages)

# ---------------- AUTH ----------------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if USERS.get(data.get("username")) == data.get("password"):
        token = create_access_token(identity=data["username"])
        return jsonify(access_token=token)
    return jsonify({"message": "Invalid credentials"}), 401

# ---------------- FILE UPLOAD ----------------
@app.route("/fileupload", methods=["POST"])
@jwt_required()
def upload_pdf():
    user = get_jwt_identity()

    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify({"message": "Valid PDF required"}), 400

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    docs = extract_text(path)
    if not docs:
        return jsonify({
            "message": "Scanned or image-based PDF detected. Text PDFs only."
        }), 400

    USER_INDEX[user] = FAISS.from_documents(docs, embeddings)
    return jsonify({"message": "PDF indexed successfully"})

# ---------------- QUERY ----------------
@app.route("/query", methods=["POST"])
@jwt_required()
def query_pdf():
    user = get_jwt_identity()
    query = request.json.get("query", "").strip()

    if user not in USER_INDEX:
        return jsonify({"message": "Upload a PDF first"}), 400

    retriever = USER_INDEX[user].as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    if not docs:
        return jsonify({"answer": "No relevant information found"})

    # Return best matching paragraph
    return jsonify({
        "answer": docs[0].page_content.strip()
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Fast PDF Search API running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

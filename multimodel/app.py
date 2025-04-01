import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import pytesseract
from PIL import Image
import google.generativeai as genai
from config import Config

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data/documents')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# RAG System
class RAGSystem:
    def __init__(self, vector_dim=128):
        self.vector_dim = vector_dim
        self.index = faiss.IndexFlatL2(vector_dim)
        self.vectorizer = TfidfVectorizer(max_features=vector_dim)
        self.documents = []
        self.vocabulary = None

    def _get_vectors(self, texts, is_query=False):
        if is_query:
            if self.vocabulary is None:
                raise ValueError("The TF-IDF vectorizer is not fitted")
            self.vectorizer.vocabulary_ = self.vocabulary
            vectors = self.vectorizer.transform(texts).toarray()
        else:
            vectors = self.vectorizer.fit_transform(texts).toarray()
            self.vocabulary = self.vectorizer.vocabulary_
        return vectors.astype('float32')

    def add_documents(self, texts):
        vectors = self._get_vectors(texts, is_query=False)
        self.documents.extend(texts)
        self.index.add(vectors)

    def search(self, query, k=3):
        query_vector = self._get_vectors([query], is_query=True)
        D, I = self.index.search(query_vector, k)
        results = [self.documents[i] for i in I[0] if i < len(self.documents)]
        return results

    def extract_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Initialize RAG system
rag_system = RAGSystem()

# Streamlit UI
st.title("Multimodal RAG System with Generative AI")

# File upload section
st.header("Upload a File (PDF or Image)")
uploaded_file = st.file_uploader("Upload a PDF or an image file (e.g., PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])

# Question input
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if uploaded_file and question:
    try:
        extracted_text = ""

        # Check if the uploaded file is a PDF
        if uploaded_file.name.lower().endswith("pdf"):
            filepath = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"PDF uploaded successfully: {uploaded_file.name}")

            # Extract text from the PDF
            extracted_text = rag_system.extract_from_pdf(filepath)
            if extracted_text.strip():
                st.success("Text extracted from the PDF.")
                # Chunk and add to FAISS index
                chunks = [extracted_text[i:i+1000] for i in range(0, len(extracted_text), 1000)]
                rag_system.add_documents(chunks)
                st.success(f"Document processed successfully with {len(chunks)} chunks.")
            else:
                st.warning("No text found in the PDF.")

        # Check if the uploaded file is an image
        elif uploaded_file.name.lower().endswith(("png", "jpg", "jpeg")):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Extract text from the image using OCR
            extracted_text = pytesseract.image_to_string(image)
            if extracted_text.strip():
                st.success("Text extracted from the image:")
                st.write(extracted_text)
            else:
                st.warning("No text found in the image.")

        # Combine the extracted text with the question and send to generative AI
        if extracted_text.strip():
            # If the file is a PDF, search for relevant contexts
            relevant_contexts = []
            if uploaded_file.name.lower().endswith("pdf"):
                relevant_contexts = rag_system.search(question, k=3)

            # Combine contexts and extracted text
            combined_context = extracted_text
            if relevant_contexts:
                combined_context += "\n\n" + "\n".join(relevant_contexts)

            # Prepare data for generative AI
            data = {
                'parts': [{'text': f"Question: {question}. Context: {combined_context}. What is the output?"}]
            }

            # Configure and call the generative AI model
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')
            generated_response = model.generate_content(data)

            # Display the response
            st.write("Generated Response:")
            st.write(generated_response.text)
            st.write("Relevant Contexts:")
            st.json(relevant_contexts)
        else:
            st.warning("No valid context to process with the question.")
    except Exception as e:
        st.error(f"Error processing file or query: {str(e)}")
elif question:
    st.warning("Please upload a file to provide context for the question.")
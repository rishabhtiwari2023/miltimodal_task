# Multimodal RAG System with Generative AI

This project implements a multimodal Retrieval-Augmented Generation (RAG) system using Streamlit. It allows users to upload either a PDF or an image file, processes the content, and generates a response to a user-provided question using Google's Generative AI (Gemini).

## Features
- **PDF Processing**: Extracts text from uploaded PDF files and uses FAISS for vector-based similarity search.
- **Image Processing**: Extracts text from uploaded images using OCR (Tesseract).
- **Generative AI**: Combines the extracted content with user questions and generates responses using Google's Generative AI.

## Requirements
- Python 3.8 or higher
- Tesseract-OCR installed on the system

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>


Install dependencies:
pip install -r requirements.txt
Install Tesseract-OCR:
sudo apt-get install tesseract-ocr
On Linux:


Instructions for Running with Docker
Build the Docker image:
docker build -t multimodal-rag .
Run the Docker container:
docker run -p 8501:8501 multimodal-rag
Open http://localhost:8501 in your browser to access the application.---

Instructions for Running with Docker
Build the Docker image:
docker build -t multimodal-rag .
Run the Docker container:
docker run -p 8501:8501 multimodal-rag
Open http://localhost:8501 in your browser to access the application.
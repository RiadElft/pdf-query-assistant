import os
import json
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from typing import Dict, List, Tuple

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page in doc:
            text += page.get_text()
        return text
    except (ValueError, RuntimeError) as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def extract_text_from_pdfs(directory: str) -> Dict[str, str]:
    """
    Extract text from all PDF files in a specified directory.
    """
    pdf_texts = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            print(f"Processing {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            if text is not None:
                pdf_texts[filename] = text
                print(f"Extracted text from {filename}")
            else:
                print(f"Failed to extract text from {filename}")
    return pdf_texts

def compute_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Compute embeddings for a list of texts using the provided model.
    """
    return model.encode(texts, show_progress_bar=True)

def find_similar_documents(query: str, 
                         embeddings: np.ndarray, 
                         paths: List[str], 
                         model: SentenceTransformer,
                         top_k: int = 5,
                         threshold: float = 0.1) -> List[Tuple[str, float]]:
    """
    Find similar documents to the query using cosine similarity of embeddings.
    Returns list of (document_path, similarity_score) tuples.
    """
    # Compute query embedding
    query_embedding = model.encode([query])[0]
    print("Query Embedding:", query_embedding)  # Debugging line
    
    # Compute cosine similarities
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    print("Similarities:", similarities)  # Debugging line
    
    # Get top k similar documents above threshold
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(paths[idx], float(similarities[idx])) 
               for idx in top_indices if similarities[idx] >= threshold]
    
    return results

def main():
    # Set up directories
    base_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_directory = os.path.join(base_directory, 'pdfs')
    output_directory = os.path.join(base_directory, 'output')
    os.makedirs(output_directory, exist_ok=True)

    # Load model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract text from PDFs
    pdf_texts = extract_text_from_pdfs(pdf_directory)
    
    # Prepare data for embedding computation
    paths = list(pdf_texts.keys())
    texts = list(pdf_texts.values())

    # Compute embeddings
    print("Computing embeddings for all documents...")
    embeddings = compute_embeddings(texts, model)

    # Save only the necessary files
    print("Saving embeddings and paths...")
    joblib.dump(embeddings, os.path.join(output_directory, 'embeddings.pkl'))
    joblib.dump(paths, os.path.join(output_directory, 'paths.pkl'))

    print("Model training and embedding computation complete.")

if __name__ == "__main__":
    main()
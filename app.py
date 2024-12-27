from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load the trained components
embeddings = joblib.load('output/embeddings.pkl')
paths = joblib.load('output/paths.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def generate_bot_response(query: str, results: list) -> str:
    num_results = len(results)
    if num_results == 0:
        return f"I couldn't find any documents matching '{query}'. Try a different search term?"
    
    # Create a more engaging response
    response = f"I found {num_results} document{'s' if num_results > 1 else ''} related to '{query}'.\n"
    
    # Add specific details about top matches
    for i, (path, score) in enumerate(results[:3], 1):
        confidence = int(score * 100)
        response += f"\n• Match #{i}: '{path}' (Confidence: {confidence}%)"
    
    return response

def find_pdf_paths(query: str, threshold: float = 0.3, top_k: int = 5):
    query_embedding = model.encode([query])[0]
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [(paths[idx], float(similarities[idx])) 
               for idx in top_indices if similarities[idx] >= threshold]
    
    return results

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, query: str = Form(...)):
    results = find_pdf_paths(query)
    bot_response = generate_bot_response(query, results)
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "query": query, 
            "pdf_paths": [path for path, _ in results],
            "bot_response": bot_response,
            "results_with_scores": results  # Pass the full results including scores
        }
    )

@app.get("/pdf/{pdf_name}")
async def get_pdf(pdf_name: str, highlight: str = None):
    pdf_path = os.path.join("pdfs", pdf_name)
    if not os.path.exists(pdf_path):
        return HTMLResponse(content="File not found", status_code=404)
    return FileResponse(pdf_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 
import pdfplumber
import re
import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- CONFIGURATION ---
PDF_PATH = "dii-summer-guidelines-april-2025.pdf" 
DB_PATH = "./chroma_db_nic"

def is_new_section(line):
    """
    Detects headers like:
    1. PURPOSE:
    4. ELIGIBILITY:
    12. TOKEN REMUNERATION:
    """
    # Regex: Start with number, dot, space, then UPPERCASE text, ending with colon
    pattern = r"^\d+\.\s+[A-Z\s/]+[:]?$" 
    return bool(re.match(pattern, line.strip()))

def parse_pdf_smart(pdf_path):
    print(f"⚙️ Parsing {pdf_path}...")
    documents = []
    current_section = "General / Introduction"
    current_buffer = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue
            
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue

                if is_new_section(line):
                    # 1. Save previous section if exists
                    if current_buffer:
                        full_text = "\n".join(current_buffer)
                        documents.append(Document(
                            page_content=full_text,
                            metadata={"source": "nic_manual", "chapter": current_section}
                        ))
                        current_buffer = [] # Reset buffer
                    
                    # 2. Update current section title
                    current_section = line.replace(":", "").strip() 
                    print(f"   📍 Found Section: {current_section}")
                else:
                    # Append normal text to current buffer
                    current_buffer.append(line)

    # Save the very last section
    if current_buffer:
        documents.append(Document(
            page_content="\n".join(current_buffer),
            metadata={"source": "nic_manual", "chapter": current_section}
        ))
    
    return documents

def ingest():
    # Handle filename variations
    global PDF_PATH
    if not os.path.exists(PDF_PATH):
        if os.path.exists("manual.pdf"):
            PDF_PATH = "manual.pdf"
        else:
            print(f"❌ Error: {PDF_PATH} not found.")
            return

    # 1. Parse
    docs = parse_pdf_smart(PDF_PATH)
    print(f"✅ Extracted {len(docs)} logical sections.")

    # 2. Embed & Store
    print("⚙️ Embedding and Storing in ChromaDB...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Force delete old DB to ensure clean slate
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print("✅ Ingestion Complete. Vector Store Ready.")

if __name__ == "__main__":
    ingest()
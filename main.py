import time
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# --- CONFIGURATION ---
DB_PATH = "./chroma_db_nic"
MODEL_NAME = "llama3.2:1b"

print("⏳ Loading System Resources...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if not os.path.exists(DB_PATH):
    print(f"❌ Error: Database not found at {DB_PATH}. Run ingest.py first!")
    exit()

vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
llm = Ollama(model=MODEL_NAME)

# Load available chapters from DB
db_data = vector_store.get()
all_chapters = list(set([m['chapter'] for m in db_data['metadatas']]))

def get_router_decision(query):
    """
    Hybrid Router (Golden Version):
    Relies on Keywords first (100% accuracy) -> Falls back to LLM.
    """
    q_lower = query.lower()
    
    # --- LEVEL 1: RULE-BASED ROUTING (Safe Mode) ---
    keyword_map = {
        # --- SECTION 1: PURPOSE ---
        "purpose": "PURPOSE",
        "aim": "PURPOSE",
        "goal": "PURPOSE",
        "objective": "PURPOSE",
        "why join": "PURPOSE",

        # --- SECTION 2: ABOUT NIC ---
        "about nic": "ABOUT NIC",
        "what is nic": "ABOUT NIC",
        "organization": "ABOUT NIC",
        "ministry": "ABOUT NIC",
        "meity": "ABOUT NIC",
        
        # --- SECTION 4: ELIGIBILITY (Who can apply) ---
        "eligib": "ELIGIBILITY",  # Covers eligibility, eligible
        "who can": "ELIGIBILITY",
        "qualif": "ELIGIBILITY", # Qualification
        "mark": "ELIGIBILITY",   # Marks
        "cgpa": "ELIGIBILITY",
        "percent": "ELIGIBILITY", # Percentage
        "b.tech": "ELIGIBILITY",
        "m.tech": "ELIGIBILITY",
        "mca": "ELIGIBILITY",
        "msc": "ELIGIBILITY",
        "student": "ELIGIBILITY",
        "degree": "ELIGIBILITY",
        "passed out": "ELIGIBILITY", # "recently passed out" mention
        
        # --- SECTION 5: DURATION (Time & Domains List) ---
        "duration": "DURATION",
        "long": "DURATION",
        "period": "DURATION",
        "month": "DURATION",
        "start": "DURATION",  # Start date
        "december": "DURATION",
        # The Domains (Mapped to Duration because the list is there)
        "domain": "DURATION",
        "area": "DURATION",
        "java": "DURATION",
        "cloud": "DURATION",
        "block": "DURATION", # Blockchain
        "chain": "DURATION",
        "ai": "DURATION",
        "artificial": "DURATION",
        "learning": "DURATION", # Machine Learning
        "data": "DURATION", # Data Analytics
        "web": "DURATION",
        "angular": "DURATION",
        "react": "DURATION",
        "php": "DURATION",
        "devops": "DURATION",
        "bot": "DURATION", # Chatbot
        "iot": "DURATION",
        "internet of things": "DURATION",
        "security": "DURATION", # Cyber Security
        "mobile": "DURATION", # Mobile App
        "app": "DURATION",
        "net": "DURATION", # .NET
        "network": "DURATION",

        # --- SECTION 6: PLACE (Location) ---
        "place": "PLACE OF INTERNSHIP",
        "location": "PLACE OF INTERNSHIP",
        "where": "PLACE OF INTERNSHIP",
        "city": "PLACE OF INTERNSHIP",
        "delhi": "PLACE OF INTERNSHIP",
        "headquarter": "PLACE OF INTERNSHIP",
        "state": "PLACE OF INTERNSHIP",
        "center": "PLACE OF INTERNSHIP",
        "centre": "PLACE OF INTERNSHIP",

        # --- SECTION 7: HOW TO APPLY ---
        "apply": "HOW TO APPLY",
        "application": "HOW TO APPLY",
        "portal": "HOW TO APPLY",
        "online": "HOW TO APPLY",
        "website": "HOW TO APPLY",
        "link": "HOW TO APPLY",
        "register": "HOW TO APPLY",
        "recommendation": "HOW TO APPLY", # Recommendation letter
        "letter": "HOW TO APPLY",
        "sop": "HOW TO APPLY",
        "statement": "HOW TO APPLY",
        "write-up": "HOW TO APPLY",

        # --- SECTION 8: SELECTION ---
        "select": "SELECTION",
        "shortlist": "SELECTION",
        "interview": "SELECTION",
        "skype": "SELECTION", # Mentioned in doc
        "merit": "SELECTION",

        # --- SECTION 9: CODE OF CONDUCT (Rules) ---
        "conduct": "CODE OF CONDUCT",
        "rule": "CODE OF CONDUCT",
        "regulat": "CODE OF CONDUCT",
        "behavior": "CODE OF CONDUCT",
        "confidential": "CODE OF CONDUCT",
        "secret": "CODE OF CONDUCT",
        "property": "CODE OF CONDUCT", # Intellectual property
        "attendance": "CODE OF CONDUCT",
        "timing": "CODE OF CONDUCT",

        # --- SECTION 10: PLACEMENT (Job Assurance) ---
        "job": "PLACEMENT",
        "employ": "PLACEMENT",
        "hired": "PLACEMENT",
        "placement": "PLACEMENT",
        "career": "PLACEMENT",
        "future": "PLACEMENT",

        # --- SECTION 11: SUBMISSION (Reports) ---
        "report": "SUBMISSION",
        "submit": "SUBMISSION",
        "presentation": "SUBMISSION",
        "project": "SUBMISSION",
        "no-demand": "SUBMISSION", # No-Demand Certificate

        # --- SECTION 12: REMUNERATION (Money) ---
        "stipend": "TOKEN REMUNERATION",
        "salary": "TOKEN REMUNERATION",
        "money": "TOKEN REMUNERATION",
        "pay": "TOKEN REMUNERATION",
        "amount": "TOKEN REMUNERATION",
        "rs.": "TOKEN REMUNERATION",
        "10000": "TOKEN REMUNERATION",
        "10,000": "TOKEN REMUNERATION",
        "fund": "TOKEN REMUNERATION",
        "remuneration": "TOKEN REMUNERATION",

        # --- SECTION 13: CERTIFICATE ---
        "certificate": "CERTIFICAT", # "CERTIFICAT" matches the PDF typo
        "certificat": "CERTIFICAT",
        "completion": "CERTIFICAT",
        
        # --- SECTION 14: TERMINATION ---
        "terminat": "TERMINATION",
        "end": "TERMINATION",
        "quit": "TERMINATION",
        "leave": "TERMINATION",
        "resign": "TERMINATION",
        "fire": "TERMINATION",
        "notice": "TERMINATION", # Notice period
        "remove": "TERMINATION"
    }

    for key, target_partial in keyword_map.items():
        if key in q_lower:
            for chapter in all_chapters:
                # Partial match to find the actual chapter name in DB
                if target_partial in chapter:
                    return chapter

    # --- LEVEL 2: LLM ROUTING (Fallback) ---
    chapters_str = ", ".join([f"'{c}'" for c in all_chapters])
    prompt = f"""
    You are a routing assistant. Pick the section name that best fits the query.
    Available Sections: [{chapters_str}]
    User Query: "{query}"
    Return ONLY the section name. If unsure, return "ALL".
    """
    try:
        response = llm.invoke(prompt).strip()
        for chapter in all_chapters:
            if chapter in response:
                return chapter
    except:
        pass
            
    return "ALL"

def run_hybrid_rag(user_query):
    print(f"\n{'='*50}")
    print(f"🗣️  User Query: {user_query}")

    # --- 1. ROUTING ---
    target_section = get_router_decision(user_query)
    
    search_kwargs = {"k": 3} 
    if target_section != "ALL":
        print(f"🎯 Router Decision: Filter Database by [chapter == '{target_section}']")
        search_kwargs["filter"] = {"chapter": target_section}
    else:
        print(f"🌐 Router Decision: Search Entire Document")

    # --- 2. RETRIEVAL ---
    t0 = time.time()
    results = vector_store.similarity_search(user_query, **search_kwargs)
    t1 = time.time()
    
    if not results:
        print("❌ No information found.")
        return

    # --- 3. GENERATION ---
    context = "\n---\n".join([doc.page_content for doc in results])
    
    prompt = f"""
    Context from Manual:
    {context}
    
    User Question: {user_query}
    
    Answer the question using ONLY the context above. 
    
    """
    
    t2 = time.time()
    answer = llm.invoke(prompt)
    t3 = time.time()

    # --- OUTPUT ---
    print(f"\n🤖 Bot Answer:\n{answer.strip()}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   - Metadata Filter: {target_section}")
    print(f"   - Retrieval Latency: {(t1-t0):.4f}s")
    print(f"   - Generation Latency: {(t3-t2):.4f}s")
    print(f"   - Total Latency:   {(t1-t0 + t3-t2):.4f}s")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    while True:
        q = input("User Query (or 'exit'): ")
        if q.lower() == "exit": break
        run_hybrid_rag(q)
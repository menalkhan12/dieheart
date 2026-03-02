import os
import re
import json
import logging
from pathlib import Path

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from groq_utils import get_client, num_keys, GROQ_KEYS
if not GROQ_KEYS:
    raise ValueError("GROQ_API_KEY or GROQ_API_KEYS not found.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
]

documents = []
doc_names = []
vectorizer = None
doc_vectors = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 12000

BASELINE_FILES = ["RAG_QUICK_REF.txt", "IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt", "ADMISSION_INFO.txt"]

KEYWORD_FILE_MAP = [
    (["closing merit", "last merit", "last year merit", "last year closing",
      "previous merit", "closing aggregate", "merit history", "merit 2024", "merit 2023"],
     ["CLOSING_MERIT_HISTORY.txt", "RAG_QUICK_REF.txt"]),

    (["fee structure", "fee of", "fees of", "fee for", "fees for", "tuition fee",
      "semester fee", "how much fee", "fee avionics", "fee aerospace", "fee electrical",
      "fee mechanical", "fee computer", "fee space", "fee mathematics", "fee physics",
      "fee biotechnology", "fee materials", "fee software", "cost of", "charges",
      "per semester", "fee per"],
     ["FEE_STRUCTURE.txt", "RAG_QUICK_REF.txt"]),

    (["programs under", "program under", "programs offered", "programs in",
      "offered by", "offered under", "department offer", "department programs",
      "what programs", "which programs", "degree programs", "all programs",
      "list of programs", "what degrees", "which degrees", "under mechanical",
      "under electrical", "under avionics", "under aerospace", "under computing",
      "under materials", "under space"],
     ["IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt", "RAG_QUICK_REF.txt",
      "ELECTRICAL_DEPARTMENT_PROGRAMS.txt"]),

    (["admission open", "admissions open", "when do admission", "last date",
      "last date to apply", "admission close", "admission deadline",
      "when to apply", "application deadline", "admission date"],
     ["ADMISSION_DATES_AND_STATUS.txt", "ADMISSION_INFO.txt"]),

    (["calculate merit", "calculate aggregate", "my aggregate", "my merit",
      "will i get admission", "aggregate formula", "merit formula",
      "merit calculation", "merit for admission", "merit calculation for",
      "regarding merit", "merit regard"],
     ["MERIT_CRITERIA_AND_AGGREGATE.txt"]),

    (["transport", "bus", "shuttle", "pick and drop", "route"],
     ["TRANSPORT_HOSTEL_FAQS.txt"]),

    (["hostel", "accommodation", "boarding", "dorm", "stay on campus"],
     ["TRANSPORT_HOSTEL_FAQS.txt", "ADMISSION_FAQS_COMPLETE.txt"]),

    (["eligible", "eligibility", "can i apply", "who can apply",
      "pre-medical", "pre medical", "ics student", "dae", "a-level", "a level"],
     ["PROGRAMS_ELIGIBILITY_WHATSAPP_SEATS.txt", "ADMISSION_FAQS_COMPLETE.txt"]),

    (["diploma", "diplomas"],
     ["PROGRAMS_ELIGIBILITY_WHATSAPP_SEATS.txt", "IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt"]),

    (["foreign", "international student", "overseas", "non-pakistani",
      "other country", "from abroad"],
     ["FOREIGN_ADMISSION.txt"]),

    (["scholarship", "financial aid", "need based", "honhaar", "peef", "fee waiver",
      "merit based scholarship", "merit scholarship"],
     ["ADMISSION_FAQS_COMPLETE.txt", "TRANSPORT_HOSTEL_FAQS.txt",
      "IST_FULL_WEBSITE_MANUAL.txt", "ADMISSION_INFO.txt"]),

    (["challan", "fee submission", "last date to submit fee", "submit fee",
      "fee deadline", "challan deadline"],
     ["ANNOUNCEMENTS.txt", "FEE_STRUCTURE.txt"]),

    # Research / labs
    (["research", "lab", "labs", "laboratory", "research group", "research center",
      "research centre", "research area", "eye vision", "space systems",
      "astronomy resource", "ncfa", "failure analysis", "remote sensing lab",
      "cubesat", "icube", "plasma", "gravitational", "spacecraft", "telescope",
      "research at ist", "ist research"],
     ["11_RESEARCH.txt", "IST_FULL_WEBSITE_MANUAL.txt", "03_ABOUT.txt"]),

    # Facilities / campus life
    (["facilit", "campus life", "gym", "sports", "cafeteria", "tuck shop",
      "wellness", "counseling", "health", "ambulance", "library"],
     ["06_FACILITIES.txt", "IST_FULL_WEBSITE_MANUAL.txt"]),

    # About IST
    (["about ist", "what is ist", "tell me about", "ist established",
      "ist history", "ist location", "where is ist", "islamabad highway",
      "ist accredited", "ist university", "space technology university"],
     ["03_ABOUT.txt", "IST_FULL_WEBSITE_MANUAL.txt"]),

    # MS / PhD
    (["ms program", "ms programs", "ms degree", "phd program", "phd degree",
      "master program", "master degree", "graduate program", "postgraduate",
      "ms admission", "phd admission", "ms fee", "phd fee", "master"],
     ["IST_DEPARTMENTS_AND_PROGRAMS_SUMMARY.txt", "FEE_STRUCTURE.txt",
      "IST_FULL_WEBSITE_MANUAL.txt"]),
]


def _chunk_text(text, max_len=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_len:
            current = (current + "\n\n" + p).strip() if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) > max_len:
                for i in range(0, len(p), max_len - overlap):
                    chunk = p[i: i + max_len].strip()
                    if chunk:
                        chunks.append(chunk)
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return chunks


def load_documents():
    global documents, doc_names
    data_folder = "data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        return
    documents = []
    doc_names = []
    try:
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            if not os.path.isfile(file_path):
                continue
            try:
                if file.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            for chunk in _chunk_text(content):
                                if chunk:
                                    documents.append(chunk)
                                    doc_names.append(file)
                            logger.info(f"Loaded: {file}")
                elif file.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        content = json.dumps(json_data, indent=2)
                        if content.strip():
                            for chunk in _chunk_text(content, max_len=1200):
                                if chunk:
                                    documents.append(chunk)
                                    doc_names.append(file)
                            logger.info(f"Loaded: {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        if documents:
            logger.info(f"Loaded {len(documents)} chunks from documents")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")


def initialize_rag():
    global vectorizer, doc_vectors
    load_documents()
    if not documents:
        logger.warning("No documents found")
        return
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, min_df=1, max_df=0.95)
        doc_vectors = vectorizer.fit_transform(documents)
        logger.info("RAG initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing RAG: {e}")
        raise


def _fix_stt_errors(text):
    replacements = {
        "mephee": "fee", "mifi": "fee", "mefi": "fee",
        "harassment": "hostel", "hostile": "hostel", "hotel": "hostel",
        "isp ": "IST ", "isd ": "IST ", "i.s.t": "IST",
        "iesp": "IST", "i.e.s.p": "IST",
    }
    lower = text.lower()
    for wrong, correct in replacements.items():
        lower = lower.replace(wrong, correct)
    return lower


def _is_end_call(q):
    return any(p in q for p in [
        "end the call", "end call", "goodbye", "bye bye", "hang up",
        "stop the call", "that's all", "thats all", "no more questions",
        "nothing else", "i'm done", "im done", "ok bye", "okay bye"
    ])


def _is_thanks_or_compliment(query):
    q = query.lower().strip()
    # Remove common filler before checking
    q_clean = re.sub(r'\b(thank you|thanks|thx|ok thanks|okay thanks|thank)\b[.,!?]?\s*', '', q).strip()
    thanks = any(x in q for x in ["thank", "thanks", "thx", "ok thanks", "okay thanks"])
    compliment = any(x in q for x in ["you're good", "youre good", "great job", "well done",
                                       "you're helpful", "youre helpful", "good job", "awesome"])
    # If there's still a real question after removing thanks phrases, don't short-circuit
    has_question = bool(q_clean) and len(q_clean) > 4
    if has_question:
        return False, False
    return thanks, compliment


def _get_forced_files_for_query(query_lower):
    forced = []
    seen = set()
    for keywords, files in KEYWORD_FILE_MAP:
        if any(kw in query_lower for kw in keywords):
            for f in files:
                if f not in seen:
                    forced.append(f)
                    seen.add(f)
    if not forced:
        logger.info(f"No keyword match, using baseline: {BASELINE_FILES}")
        for f in BASELINE_FILES:
            if f not in seen:
                forced.append(f)
                seen.add(f)
    return forced


def _get_chunks_from_files(filenames):
    file_set = set(filenames)
    parts = []
    total = 0
    for idx, name in enumerate(doc_names):
        if name in file_set:
            chunk = f"[{name}]\n{documents[idx]}"
            if total + len(chunk) <= MAX_CONTEXT_CHARS - 1000:
                parts.append(chunk)
                total += len(chunk)
    return "\n\n---\n\n".join(parts) if parts else ""


def _expand_query_for_retrieval(query):
    q = query.lower()
    extra = []
    if any(w in q for w in ["cost", "price", "tuition", "fee", "fees"]):
        extra.append("fee structure tuition semester charges rupees lakh thousand")
    if any(w in q for w in ["merit", "closing", "aggregate", "calculate", "last year"]):
        extra.append("merit aggregate closing 2024 2023 Computer Science Space Science engineering")
    if any(w in q for w in ["research", "lab", "laboratory"]):
        extra.append("research lab Space Systems Astronomy CubeSat NCFA remote sensing telescope")
    if extra:
        return query + " " + " ".join(extra)
    return query


def retrieve_context(query, top_k=12):
    global vectorizer, doc_vectors
    if vectorizer is None or doc_vectors is None:
        return ""
    try:
        q_lower = query.lower()
        forced_files = _get_forced_files_for_query(q_lower)
        forced_context = _get_chunks_from_files(forced_files)
        logger.info(f"Forced files injected: {forced_files}")

        expanded = _expand_query_for_retrieval(query)
        query_vec = vectorizer.transform([expanded])
        similarities = cosine_similarity(query_vec, doc_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        forced_file_set = set(forced_files)
        tfidf_parts = []
        total_len = len(forced_context)

        for idx in top_indices:
            if similarities[idx] <= 0.005:
                continue
            if doc_names[idx] in forced_file_set:
                continue
            chunk = f"[{doc_names[idx]}]\n{documents[idx]}"
            if total_len + len(chunk) > MAX_CONTEXT_CHARS:
                remain = MAX_CONTEXT_CHARS - total_len - 80
                if remain > 400:
                    tfidf_parts.append(chunk[:remain] + "\n...[truncated]")
                break
            tfidf_parts.append(chunk)
            total_len += len(chunk)

        tfidf_context = "\n\n---\n\n".join(tfidf_parts)
        final_context = (forced_context + "\n\n---\n\n" + tfidf_context) if (forced_context and tfidf_context) else (forced_context or tfidf_context)
        logger.info(f"Context length: {len(final_context)} chars, forced_files: {forced_files}")
        return final_context
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""


def generate_answer(query, conversation_history=None):
    query = _fix_stt_errors(query)
    q_lower = query.lower().strip()

    if _is_end_call(q_lower):
        return ("Thank you for calling IST. Have a great day! Goodbye.", False)

    is_thanks, is_compliment = _is_thanks_or_compliment(query)
    if is_thanks:
        return ("You're welcome.", False)
    if is_compliment:
        return ("Thank you.", False)

    user_message = query
    retrieval_query = query
    if conversation_history:
        hist_str = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversation_history])
        user_message = f"Previous conversation:\n{hist_str}\n\nCurrent query: {query}"
        last_user, last_agent = conversation_history[-1] if conversation_history else ("", "")
        if any(w in query.lower() for w in ["its", "their", "them", "they", "that", "those", "it", "this", "same"]):
            retrieval_query = f"{query} {last_user} {last_agent}"

    context = retrieve_context(retrieval_query)

    if not context.strip():
        return ("I don't have that information. Please provide your phone number and we will contact you.", True)

    system_prompt = f"""You are the official voice assistant for Institute of Space Technology (IST). You answer callers by phone.

AUTHORITATIVE PROGRAMS BY DEPARTMENT:
- Aeronautics and Astronautics: BS Aerospace Engineering only
- Avionics Engineering: BS Avionics Engineering only
- Electrical Engineering: BS Electrical Engineering and BS Computer Engineering (ONLY these two)
- Mechanical Engineering: BS Mechanical Engineering (ONLY this one)
- Materials Science and Engineering: BS Materials Science and Engineering and BS Biotechnology
- Space Science: BS Space Science and BS Physics
- Computing: BS Computer Science, BS Software Engineering, BS Data Science, BS Artificial Intelligence
- Applied Mathematics and Statistics: BS Mathematics only
- MS programs: Aerospace, Electrical, Materials, Mechanical, Computer Science, Mathematics, Physics, Astronomy
- PhD programs: Aerospace, Electrical, Materials, Mathematics, Physics, Astronomy

RULES:
1. Answer ONLY from CONTEXT. NEVER invent. NEVER say "check the website".
2. Keep response to MAX 2 short sentences. Be concise for phone conversation.
3. DIPLOMA: IST does NOT offer diplomas. IST accepts DAE holders as applicants only.
4. RESEARCH LABS: IST has Space Systems Lab, Astronomy Resource Center (16-inch telescope), NCFA (failure analysis), NCRS&GI (remote sensing). Answer from CONTEXT.
5. FEE: Use lakh and thousand format. Give specific program fee when asked.
6. CLOSING MERIT: Last year = 2024. Give number directly.
7. STAY ON TOPIC: Only answer what the user specifically asked. Do NOT volunteer unrelated information from context (e.g. if asked about scholarships, do NOT mention closing merit numbers).
8. If truly not in CONTEXT: "I don't have that information. Please provide your phone number and we will contact you."

CONTEXT:
{context}"""

    for key_idx in range(num_keys()):
        client = get_client(key_idx)
        for model in MODELS:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=120  # Short = faster TTS + better iOS playback
                )
                reply = response.choices[0].message.content
                if not reply or not reply.strip():
                    continue
                reply = reply.strip()
                logger.info(f"LLM reply from {model}: {reply}")
                escalated = any(p in reply.lower() for p in [
                    "technical issue", "cannot find", "unable",
                    "phone number", "provide your phone", "contact you"
                ])
                return reply, escalated
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited on {model}, rotating key...")
                    break  # Immediately try next key
                if "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    logger.warning(f"Key {key_idx+1} invalid, rotating key...")
                    break
                logger.error(f"Model {model} key {key_idx+1} error: {e}")
                continue

    logger.error("All keys/models exhausted")
    return ("I'm having technical difficulties. Please provide your phone number and we will call you back.", True)
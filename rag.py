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

from groq_utils import get_client, num_keys, get_next_key_index, GROQ_KEYS
if not GROQ_KEYS:
    import warnings
    warnings.warn("GROQ_API_KEY or GROQ_API_KEYS not set. RAG will fail until configured on Render.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

MODELS = ["llama-3.1-8b-instant"]  # Use only fast model for 1-2 sec response

documents = []
doc_names = []
vectorizer = None
doc_vectors = None

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 8000

BASELINE_FILES = ["all_content.txt", "calling_assistant_kb.json"]

# All file names refer to files in ist_output folder (KB). Same rules: answer only from CONTEXT.
KEYWORD_FILE_MAP = [
    # Broad admission FAQs - answer directly, do not say "check website"
    (["admission faq", "admission question", "admission query", "spring admission", "fall admission",
      "where to apply", "apply online", "admission portal", "last date to apply", "deadline",
      "age limit", "how many programs", "apply via post", "track application", "arn", "miss deadline",
      "entry test", "nat", "hat", "net", "ecat", "etea", "nts", "which test", "sat score", "negative marking",
      "documents", "equivalence", "ibcc", "cnic", "attested", "domicile", "character certificate", "migration",
      "merit list", "closing merit", "waiting list", "confirm seat", "freeze semester", "classes start",
      "orientation", "grading", "gpa", "cgpa", "admissions office", "9075406", "9075100",
      "application fee", "processing fee", "aerospace vs avionics", "avionics vs aerospace",
      "materials science", "space science different", "supply in fsc", "provisional admission",
      "visit campus", "digital process", "self finance", "refundable", "installments", "endowment",
      "hostel fee", "hostel mess", "laundry", "wifi", "sports", "gym", "career", "job fair", "cdc",
      "washington accord", "pec accredited", "medium of instruction", "ist under military"],
     ["admission_faqs_complete.txt", "all_content.txt", "calling_assistant_kb.json"]),

    (["closing merit", "last merit", "last year merit", "closing aggregate", "merit history", "merit 2024", "merit 2023",
      "calculate aggregate", "calculate my aggregate", "merit formula", "will merit", "will i get admission",
      "aggregate for", "merit criteria", "matric marks", "fsc marks", "entry test marks",
      "merit determination", "determination criteria", "merit of", "engineering merit", "computing merit"],
     ["admission_faqs_complete.txt", "merit_faq.txt", "all_content.txt", "calling_assistant_kb.json", "programs.csv"]),

    (["fee structure", "fee of", "fees of", "fee for", "fees for", "tuition fee", "semester fee", "how much fee",
      "cost of", "charges", "per semester", "fee per", "challan", "fee submission", "fee deadline",
      "bs computer science fee", "bs avionics fee", "computer science fee", "avionics fee", "fee of bs",
      "ms fee", "ms computer science fee", "fee of ms", "graduate fee", "phd fee", "application processing fee",
      "admission fee", "hidden charges", "refund", "summer semester fee", "international fee",
      " fee", " fees", "id card", "freeze charges", "freeze fee", "semester charges"],
     ["admission_faqs_complete.txt", "fee_faq.txt", "calling_assistant_kb.json", "all_content.txt", "programs.csv"]),

    (["hod of", "head of department", "who is the hod", "who is the head of", "head of electrical",
      "head of avionics", "head of computing", "hod of electrical", "hod of avionics", "department of computing"],
     ["hod_faq.txt", "faculty.csv", "contacts.csv", "calling_assistant_kb.json"]),

    (["programs under", "program under", "programs offered", "programs in", "offered by", "department programs",
      "what programs", "which programs", "degree programs", "all programs", "list of programs",
      "computing department", "computing programs", "undergraduate computing", "programs under computing",
      "computing undergraduate", "bs computing", "computing degree", "electrical department", "space science department",
      "computer electrical space science", "programs in computing", "programs in electrical"],
     ["programs_faq.txt", "programs.csv", "all_content.txt", "calling_assistant_kb.json"]),

    (["admission open", "admissions open", "when do admission", "last date", "admission close", "admission deadline",
      "when to apply", "application deadline", "admission date", "eligible", "eligibility", "can i apply", "who can apply",
      "pre-medical", "pre medical", "ics student", "dae", "a-level", "criteria for", "requirements for program",
      "biotechnology", "fsc pre-engineering", "pre-engineering", "can i apply for", "apply in bs"],
     ["admission_faqs_complete.txt", "eligibility_faq.txt", "programs.csv", "all_content.txt", "calling_assistant_kb.json"]),

    (["transport", "bus", "shuttle", "pick and drop", "route", "hostel", "accommodation", "boarding", "dorm",
      "transport to students", "does ist offer transport", "offer transport", "hostel fee", "03000544707"],
     ["admission_faqs_complete.txt", "transport_faq.txt", "calling_assistant_kb.json", "all_content.txt"]),

    (["vice chancellor", "vc ", " vc", "who is the vc", "who is vice chancellor"],
     ["vc_faq.txt", "faculty.csv", "contacts.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["vc message", "vc's message", "vice chancellor message", "message of vc", "vision of ist", "ist vision"],
     ["all_content.txt", "calling_assistant_kb.json"]),

    (["harassment", "harassment policy", "sexual harassment", "complaint cell", "hcc",
      "complaint policy", "how to complain", "file complaint", "harassment complaint"],
     ["harassment_faq.txt", "all_content.txt", "calling_assistant_kb.json"]),

    (["mess", "mess facility", "cafeteria", "canteen", "food", "dining"],
     ["all_content.txt", "calling_assistant_kb.json"]),

    (["faculty", "professor", "dr.", "doctor", "lecturer", "registrar", "dean",
      "teacher", "instructor", "contact person"],
     ["faculty.csv", "contacts.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["contact", "phone", "address", "email", "how to reach", "where is ist", "location",
      "location of ist", "where is ist located", "ist address", "ist location", "driving directions",
      "faizabad", "islamabad highway", "cda toll"],
     ["contacts.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["recent events", "upcoming workshops", "upcoming events", "current workshops", "workshops at ist"],
     ["news.csv", "programs.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["timings", "timing", "office hours", "working hours", "when is ist open", "ist open", "opening hours"],
     ["office_timings_faq.txt", "contacts.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["procedure to apply", "how to apply", "application process", "steps to apply", "apply in ist", "admission process"],
     ["admission_faqs_complete.txt", "eligibility_faq.txt", "programs.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["research", "lab", "labs", "laboratory", "research center", "research centres", "research centers",
      "cubesat", "icube", "astronomy", "ncfa", "ncgsa", "national center of gis", "gis and space applications",
      "lunar mission", "lunar", "moon mission", "icube-q", "icube qamar", "chang'e", "pakistan moon",
      "oric", "bic", "cset", "arc", "ssl", "cisl", "wisp"],
     ["research_centres_faq.txt", "suparco_faq.txt", "ncgsa_faq.txt", "calling_assistant_kb.json", "all_content.txt", "news.csv"]),

    (["qamar ul islam", "kamarul islam", "dr qamar", "dr. qamar", "qamar islam"],
     ["faculty.csv", "calling_assistant_kb.json", "all_content.txt"]),

    (["director of", "who is the director", "ncfa director", "failure analysis director",
      "national center for failure", "director ncfa", "head of ncfa", "who heads ncfa",
      "founder of ncfa", "when was ncfa founded", "ncfa founded", "ncfa established",
      "founder of ncga", "founder of ncgsa", "ncga founder", "ncgsa founder"],
     ["ncfa_director.txt", "ncgsa_faq.txt", "calling_assistant_kb.json", "all_content.txt"]),

    (["about ist", "what is ist", "tell me about", "ist established", "ist history", "vision", "mission"],
     ["all_content.txt", "calling_assistant_kb.json"]),

    (["campus life", "facilities", "gym", "sports", "cafeteria", "wellness", "counseling", "health"],
     ["all_content.txt", "calling_assistant_kb.json"]),

    (["news", "events", "convocation", "workshop", "conference", "icast", "iceast", "job fair"],
     ["news.csv", "programs.csv", "all_content.txt", "calling_assistant_kb.json"]),

    (["quality assessment", "quality assessments", "qec", "assessment 2012", "program teams 2012", "quality 2012"],
     ["quality_2012_faq.txt", "all_content.txt", "news.csv", "programs.csv"]),

    (["scholarship", "financial aid", "foreign", "international student", "ms program", "phd program", "master",
      "need based", "merit scholarship", "honhaar", "peef", "suparco sponsor", "financial aid office"],
     ["admission_faqs_complete.txt", "all_content.txt", "calling_assistant_kb.json", "programs.csv"]),

    (["sparco", "suparco", "space agency", "pakistan space"],
     ["suparco_faq.txt", "all_content.txt", "calling_assistant_kb.json"]),
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
    data_folder = "ist_output"
    if not os.path.exists(data_folder):
        logger.warning(f"KB folder not found: {data_folder}")
        return
    documents = []
    doc_names = []
    try:
        for file in sorted(os.listdir(data_folder)):
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
                    if file.lower() == "raw_api_data.json":
                        continue
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        content = json.dumps(json_data, indent=2)
                        if content.strip():
                            for chunk in _chunk_text(content, max_len=1200):
                                if chunk:
                                    documents.append(chunk)
                                    doc_names.append(file)
                            logger.info(f"Loaded: {file}")
                elif file.endswith(".csv"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if content.strip():
                            for chunk in _chunk_text(content, max_len=1000):
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


def _fix_stt_errors(text):
    replacements = {
        "mephee": "fee", "mifi": "fee", "mefi": "fee",
        "hostile": "hostel", "hotel": "hostel",
        "isp ": "IST ", "isd ": "IST ", "i.s.t": "IST",
        "iesp": "IST", "i.e.s.p": "IST", " isp ": " IST ", " in isp": " in IST",
        "metric ": "matric ", " metric": " matric",
        "kamarul islam": "qamar ul islam", "kamar ul islam": "qamar ul islam",
    }
    lower = text.lower()
    for wrong, correct in replacements.items():
        lower = lower.replace(wrong, correct)
    return lower


def _is_end_call(q):
    # Exact phrase matches
    exact = [
        "end the call", "end call", "goodbye", "bye bye", "hang up",
        "stop the call", "that's all", "thats all", "no more questions",
        "nothing else", "i'm done", "im done", "ok bye", "okay bye",
        "thank you bye", "thanks bye", "bye for now", "that will be all",
        "that is all", "no questions", "no further questions",
        "i am done", "we are done", "all done", "good bye",
        "have a good day", "have a nice day", "take care",
        "disconnect", "close the call", "finish the call",
        "end this call", "stop call",
    ]
    if any(p in q for p in exact):
        return True

    # Short utterances that are almost certainly goodbye
    # e.g. just "bye", "goodbye", "thanks bye"
    words = q.strip().split()
    if len(words) <= 3 and any(w in ["bye", "goodbye", "ciao", "done", "finished"] for w in words):
        return True

    return False


def _is_thanks_or_compliment(query):
    q = query.lower().strip()
    thanks = any(x in q for x in ["thank", "thanks", "thx", "ok thanks", "okay thanks"])
    compliment = any(x in q for x in ["you're good", "youre good", "great job", "well done",
                                       "you're helpful", "youre helpful", "good job", "awesome"])
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


def _strip_urls(text: str) -> str:
    """Remove URLs from reply so we never speak them over the phone."""
    if not text or not text.strip():
        return text
    out = re.sub(r'https?://[^\s\]\)\}]+', '', text, flags=re.IGNORECASE)
    out = re.sub(r'\bist\.edu\.pk[^\s\]\)\}]*', '', out, flags=re.IGNORECASE)
    # Remove "check website" phrases (model sometimes ignores prompt)
    out = re.sub(r'[.;]?\s*(However,?\s+)?(the fee structure|more information|details?)\s+(is )?(available|can be found)\s+on\s+(the )?IST\s+website\.?', '', out, flags=re.IGNORECASE)
    out = re.sub(r'[.;]?\s*Please\s+check\s+(the )?website\.?', '', out, flags=re.IGNORECASE)
    out = re.sub(r'\s+\s+', ' ', out).strip()
    return out if out else text.strip()


def _get_chunks_from_files(filenames):
    """Build context from requested files. Iterate in filenames order so priority files (e.g. faculty.csv) get in first."""
    file_set = set(filenames)
    # Collect (doc_names idx, chunk) for each requested file, in the order files were requested
    by_file = {f: [] for f in filenames if f in file_set}
    for idx, name in enumerate(doc_names):
        if name in file_set and name in by_file:
            by_file[name].append((idx, f"[{name}]\n{documents[idx]}"))
    parts = []
    total = 0
    for f in filenames:
        if f not in by_file:
            continue
        for idx, chunk in by_file[f]:
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


def retrieve_context(query, top_k=5):
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
    if not GROQ_KEYS:
        return ("Service is being configured. Please try again in a moment.", False)
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
        # Only merge previous turn into retrieval when query clearly references it (e.g. "what about that?", "same for X")
        if any(w in query.lower() for w in ["that", "those", "same", "also", "too", "what about"]):
            retrieval_query = f"{query} {last_user} {last_agent}"

    context = retrieve_context(retrieval_query)

    # Escalate if context is empty or too weak to be relevant
    if not context.strip() or len(context.strip()) < 200:
        logger.info(f"Context too weak ({len(context.strip())} chars) — escalating")
        return ("I don't have that information. Please provide your phone number and we will contact you.", True)

    system_prompt = f"""You are the official voice assistant for Institute of Space Technology (IST). You answer callers by phone.

CRITICAL RULES — FOLLOW EXACTLY:
0. NO HALLUCINATION: You MUST only state facts that appear WORD-FOR-WORD or explicitly in CONTEXT. NEVER invent, assume, or add: phone numbers, routes, program names, fees, names, or any detail. If CONTEXT does not mention it, do NOT say it. If asked for routes or a phone number and CONTEXT has none, say you don't have that. Be strict.
1. You can ONLY use information that appears directly in the CONTEXT section below. The CONTEXT is the complete knowledge base (KB) for IST.
2. If the answer is not clearly and explicitly present in CONTEXT, reply EXACTLY with this sentence and nothing else:
   "I don't have that information. Please provide your phone number and we will contact you."
3. NEVER say "check the website", "available on website", "visit the website", "visit IST website", or similar. Give the answer from CONTEXT only. Never explain, apologize, or say "I'm not sure".
4. Never answer questions about topics not in CONTEXT — use the exact escalation sentence.
5. ADMISSION FAQs: Answer ALL admission, eligibility, entry test, fee, scholarship, documents, hostel, merit-list, and campus-life questions DIRECTLY from CONTEXT. Do NOT say "check the website" or "call admissions" as the main response. Give the factual answer from CONTEXT first. If CONTEXT has it, say it.
6. For ELIGIBILITY queries (e.g. "eligibility for X", "who can apply", "criteria for program"): Give the COMPLETE eligibility/criteria from CONTEXT. Do NOT shorten; provide the full info so the caller hears everything.
7. For HARASSMENT, SEXUAL HARASSMENT, HCC, COMPLAINT CELL, COMPLAINT POLICY, VC MESSAGE, VICE CHANCELLOR MESSAGE: Give ALL important information from CONTEXT. Do NOT give just one line. Include: zero-tolerance policy, when HCC was established (Feb 2012), contact (Dr. Rahila Naz, Head of HCC, Room 212 Block 2, phone 051-9075562, email head.hcc@ist.edu.pk), that process is confidential and fair, no reprimand for legitimate complaints. For VC message include the vision, key points about IST, faculty, students, entrepreneurial focus. Speak 4-6 clear sentences so the caller hears the full important info.
8. For all other answers: Keep to maximum 2 very short sentences unless CONTEXT clearly has more detail to share. Be concise for phone.
8. Use amounts in lakh and thousand when CONTEXT gives numbers. Stay on topic; only answer what the user asked.
9. For "who is the VC/vice chancellor/HOD/head of [department]": Match the EXACT department the user asked for. If they ask "HOD of Electrical", answer ONLY the person for Electrical Engineering, NOT Computing or any other department. If CONTEXT lists multiple departments (e.g. Electrical: Dr. Adnan Zafar, Avionics: Dr. Israr Hussain, Computing: Khurram Khurshid), pick the one that matches the user's department name. Do not say you don't have the information if CONTEXT lists that person.
   VICE CHANCELLOR of IST: The ONLY correct answer is Dr. Syed Najeeb Ahmad (Maj Gen Dr. Syed Najeeb Ahmad Retd). NEVER say "Dr. Raza ibn Abubakr" or any other name for VC — that is wrong.
10. NEVER include, say, or read aloud any URL (http, https, ist.edu.pk, or any web link). Your reply is spoken over the phone — give only factual content: program names, numbers, descriptions. If CONTEXT contains both program names and URLs, list ONLY the program names (e.g. Computer Science, Artificial Intelligence, Data Science, Software Engineering, Computer Engineering). Do not mention or read any link.
11. For "who is the director of [center/unit name]" (e.g. NCFA, National Center for Failure Analysis): If CONTEXT lists "The Team" or a name with title "Director" for that center, answer with that name and "Director" from CONTEXT. Do not say you don't have the information if CONTEXT lists that person.
12. For FEE queries (fee structure, fee of X program, tuition, cost, how much, ID card, freeze): If CONTEXT has the specific fee (e.g. ID card Rs 1,000, freeze policy, Biotechnology Rs 133,289), state it. If the user asks about a specific charge (freeze, ID card) and CONTEXT does NOT mention it, use the escalation sentence. Never substitute a different fee (e.g. admission charges) for the one asked.
13. For "who is the director of [center]" (e.g. NCFA): If CONTEXT lists a name followed by "Director" in a Team section (e.g. "Dr Anjum Tauqir Director"), that person is the Director — answer with that name. Do not say director is not mentioned if CONTEXT clearly lists it.
13. For transport/bus/shuttle: If CONTEXT has transport info (e.g. pick-and-drop for Rawalpindi/Islamabad, route-details number 03000544707, or Admissions 051-9075406 / 051-9075100), say it. NEVER add numbers not in CONTEXT.
14. For NCGSA: If CONTEXT has "National Center of GIS and Space Applications" or "NCGSA", answer that NCGSA is HEC's National Center of GIS and Space Applications; IST collaborates with it.
15. For eligibility (e.g. "can I apply for BS Biotechnology with FSC pre-engineering"): If CONTEXT says Biotechnology needs "FSc in any science group with Biology, Chemistry, Physics, Mathematics, or Computer Science", FSC pre-engineering qualifies. Answer yes.
16. For quality assessment/QEC/2012: If CONTEXT has QEC, Program Teams 2012, or assessments in 2012, summarize from CONTEXT.
17. For MS/PhD fee (e.g. "fee of MS Computer Science"): MS and PhD programs share a common fee structure. If CONTEXT has "Fee Structure - MS & PhD" or "Tuition Fee" with Rs 80,526, use that. Do not say not explicitly mentioned if CONTEXT has MS fee figures.
18. For SUPARCO/SPARCO: SUPARCO is Pakistan's national space agency (Pakistan Space and Upper Atmosphere Research Commission). IST collaborates with SUPARCO (e.g. ICUBE-Qamar). If user says SPARCO, they likely mean SUPARCO. Answer from CONTEXT.
18b. NCFA (National Center for Failure Analysis): IST established it in March 2013 (as Failure Analysis Center, later upgraded to NCFA). Dr Anjum Tauqir is the Director. If asked who founded NCFA, say IST established it; no named individual founder is in CONTEXT.
18c. NCGSA / NCGA: NCGSA is HEC's National Center of GIS and Space Applications. If asked founder of NCGSA or NCGA, CONTEXT does not list a founder — say you don't have that information, or that it is an HEC center.
19. MERIT/AGGREGATE: Use formulas from CONTEXT. Need ONLY Matric total/1100 and FSC total/1100 (never subject-wise). Engineering: also Entry Test/100. Formula: Engineering = (Matric/1100×10)+(FSC/1100×40)+(Entry/100×50). Non-engineering = (Matric/1100×50)+(FSC/1100×50). If caller gave marks in this or previous message, CALCULATE and say "Your aggregate is about X." If not, ask for marks (Engineering: Matric, FSC, Entry; Non-eng: Matric, FSC only). End with: "Be hopeful; check this year's merit when the merit list is displayed on the portal."
20. CLOSING MERIT: Use historical data from CONTEXT. For "will merit increase/decrease": trend has been stable/slightly rising; exact closing known when merit list published. For Biotechnology: first year, no previous closing merit. Do not promise cutoffs.
21. PROGRAMS BY DEPARTMENT: List ONLY programs that appear in CONTEXT. Computing: Computer Science, Artificial Intelligence, Data Science, Software Engineering, Computer Engineering. Electrical: Electrical Engineering. Space Science: Space Science. Engineering: Aerospace, Avionics, Electrical, Mechanical, Metallurgy & Materials. Science: Space Science, Physics, Math, RS&GIS, Chemistry (Nanotechnology), Biotechnology. Do NOT add or remove any program. If CONTEXT lists different names, use those exact names.
22. Answer ONLY the CURRENT question. Do not mix up topics: if asked about harassment policy, answer only harassment/HCC; if asked about mess/cafeteria, answer only dining/cafeteria from CONTEXT; if asked about hostel, answer only hostel. Never give hostel info when asked about harassment, or transport when asked about mess.
23. HARASSMENT/SEXUAL HARASSMENT/HCC/COMPLAINT: Give COMPLETE important info. IST has zero-tolerance. HCC established 20 Feb 2012. Contact Dr. Rahila Naz, Head of HCC, Room 212 Block 2, 051-9075562, head.hcc@ist.edu.pk. Process is confidential and fair. No reprimand for legitimate complaints. Do NOT give just one sentence.
24. MESS/CAFETERIA/CANTEEN/DINING: If CONTEXT mentions cafeteria, dining timings, canteen, cafeteria contractor, or mess facilities, answer from that. IST has cafeteria on campus, dining timings, and cafeteria services.
25. OFFICE TIMINGS: If CONTEXT says IST is open 8 AM to 4 PM Monday to Friday, or similar office hours, state that. IST main campus and offices: 8 AM to 4 PM, Monday to Friday; closed weekends.
26. PROCEDURE TO APPLY / HOW TO APPLY: Steps from CONTEXT: (1) Admissions announced on IST website/newspapers (BS: March/April; MS/PhD: April-May and Nov-Dec), (2) Create account at www.ist.edu.pk or eportal.ist.edu.pk, complete online form, upload documents, (3) Deposit application fee via challan at Meezan Bank/HBL, (4) Merit list displayed. Give this from CONTEXT.
27. DR. QAMAR UL ISLAM (also Kamarul Islam): Professor, Electrical Engineering; PhD University of Surrey; specialization Satellite Communication/Space Systems; Project Director ICUBE-Q (lunar CubeSat), ICUBE-2; Editor-in-Chief Journal of Space Technology; Phone 051-9075428, qamar.islam@ist.edu.pk. Answer from CONTEXT.
28. LUNAR MISSION / ICUBE-Q: Pakistan's lunar CubeSat ICUBE-Q (ICUBE-Qamar) onboard China's Chang'e 6 mission; launched May 2024; IST + SUPARCO + Shanghai Jiao Tong; Dr Qamar ul Islam Project Director. ICUBE-1 launched 2013. Summarize from CONTEXT.
29. RESEARCH CENTRES / LABS: IST has ORIC, ICUBE-Q, CSET, BIC, NCFA, Astronomy Resource Center, Space Systems Lab, Cyber and Information Security Lab, AI and Computer Vision Lab, Propulsion Lab, Aerodynamics Lab, Control Systems Lab, WiSP Lab, and more. Answer from CONTEXT.
30. LOCATION / ADDRESS: IST is at 1, Islamabad Highway, Near CDA Toll Plaza, Islamabad 44000, Pakistan. About 20 min from Islamabad and Rawalpindi. Driving directions in CONTEXT.
31. EVENTS / WORKSHOPS: Convocation, ICAST, ICEAST, workshops (Plasma Spray, Python, Bridge the Gap, Lunar Satellite, etc.), Job Fair, TEDx. Answer from news.csv and all_content.

CONTEXT:
{context}"""

    first_key = get_next_key_index()
    key_order = [first_key] + [i for i in range(num_keys()) if i != first_key]
    for key_idx in key_order:
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
                    max_tokens=250
                )
                reply = response.choices[0].message.content
                if not reply or not reply.strip():
                    continue
                reply = reply.strip()
                reply = _strip_urls(reply)
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


def generate_answer_stream(query, conversation_history=None):
    """
    Stream LLM output sentence-by-sentence for low-latency TTS.
    Yields (sentence, escalated) tuples. escalated is None until the final yield.
    """
    query = _fix_stt_errors(query)
    q_lower = query.lower().strip()

    if _is_end_call(q_lower):
        reply = "Thank you for calling IST. Have a great day! Goodbye."
        yield reply, False
        return

    is_thanks, is_compliment = _is_thanks_or_compliment(query)
    if is_thanks:
        yield "You're welcome.", False
        return
    if is_compliment:
        yield "Thank you.", False
        return

    user_message = query
    retrieval_query = query
    if conversation_history:
        hist_str = "\n".join([f"User: {u}\nAgent: {a}" for u, a in conversation_history])
        user_message = f"Previous conversation:\n{hist_str}\n\nCurrent query: {query}"
        last_user, last_agent = conversation_history[-1] if conversation_history else ("", "")
        if any(w in query.lower() for w in ["that", "those", "same", "also", "too", "what about"]):
            retrieval_query = f"{query} {last_user} {last_agent}"

    context = retrieve_context(retrieval_query)

    if not context.strip() or len(context.strip()) < 200:
        logger.info(f"Context too weak ({len(context.strip())} chars) — escalating")
        yield "I don't have that information. Please provide your phone number and we will contact you.", True
        return

    system_prompt = f"""You are IST voice assistant. Only facts from CONTEXT. No URLs.
If not in CONTEXT: "I don't have that information. Please provide your phone number and we will contact you."
For HARASSMENT, HCC, COMPLAINT POLICY, VC MESSAGE: Give ALL important info (4-6 sentences). Zero-tolerance, HCC contact Dr. Rahila Naz Room 212 Block 2, 051-9075562, head.hcc@ist.edu.pk, confidential process. For VC message: vision, faculty, students, entrepreneurial focus.
For other topics: 1-2 short sentences. VC: Dr. Syed Najeeb Ahmad. HOD: match dept. Fees: from CONTEXT.

CONTEXT:
{context}"""

    first_key = get_next_key_index()
    key_order = [first_key] + [i for i in range(num_keys()) if i != first_key]
    for key_idx in key_order:
        client = get_client(key_idx)
        for model in MODELS:
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=180,
                    stream=True,
                )
                buffer = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta or not getattr(delta, "content", None):
                        continue
                    buffer += delta.content
                    parts = re.split(r'(?<=[.!?])\s+', buffer, maxsplit=1)
                    if len(parts) > 1:
                        first_sent = parts[0].strip()
                        if first_sent:
                            yield _strip_urls(first_sent), None
                        buffer = parts[1]
                    elif parts[0].strip() and parts[0].strip()[-1] in ".!?":
                        yield _strip_urls(parts[0].strip()), None
                        buffer = ""

                if buffer.strip():
                    last = _strip_urls(buffer.strip())
                    escalated = any(p in last.lower() for p in [
                        "technical issue", "cannot find", "unable",
                        "phone number", "provide your phone", "contact you"
                    ])
                    yield last, escalated
                return
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    logger.warning(f"Key {key_idx+1} rate limited on {model}, rotating key...")
                    break
                if "401" in err_str or "invalid" in err_str or "unauthorized" in err_str:
                    logger.warning(f"Key {key_idx+1} invalid, rotating key...")
                    break
                logger.error(f"Model {model} key {key_idx+1} error: {e}")
                continue

    yield "I'm having technical difficulties. Please provide your phone number and we will call you back.", True
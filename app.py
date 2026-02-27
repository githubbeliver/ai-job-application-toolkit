import os
import re
import json
import shutil
from collections import Counter

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract


# =========================
# Config / Secrets / Client
# =========================
st.set_page_config(page_title="AI Job Application Toolkit", page_icon="🧰", layout="wide")

load_dotenv()

def get_api_key() -> str | None:
    # 1) Streamlit Cloud secrets (recommended for deployment)
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # 2) Local .env
    return os.getenv("OPENAI_API_KEY")

API_KEY = get_api_key()
client = OpenAI(api_key=API_KEY) if API_KEY else None


# =========================
# OCR Config (Windows vs Cloud)
# =========================
def configure_ocr():
    """
    Configure Tesseract path depending on OS.
    - Windows: tries default install path, then PATH
    - Linux (Streamlit Cloud): uses `tesseract` from PATH
    """
    if os.name == "nt":
        win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(win_path):
            pytesseract.pytesseract.tesseract_cmd = win_path
            return win_path
        exe = shutil.which("tesseract")
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
            return exe
        return None
    else:
        exe = shutil.which("tesseract")
        if exe:
            pytesseract.pytesseract.tesseract_cmd = exe
            return exe
        return None

TESSERACT_CMD = configure_ocr()


# =========================
# Helpers
# =========================
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def extract_text_pypdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io_bytes := __import__("io").BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return clean_text("\n".join(parts))
    except Exception:
        return ""

def extract_text_ocr(pdf_bytes: bytes) -> tuple[str, str]:
    """
    Returns: (text, status_message)
    status_message used for UI (ex: "OCR used" / errors)
    """
    # Poppler is required for pdf2image on Windows, and also needed on many environments.
    # On Streamlit Cloud, poppler-utils is installed via packages.txt.
    if not TESSERACT_CMD:
        return "", "OCR failed: Tesseract not found. (On Streamlit Cloud, add packages.txt with tesseract-ocr + poppler-utils.)"

    try:
        images = convert_from_bytes(pdf_bytes)  # uses poppler internally
    except Exception as e:
        return "", f"OCR failed: Unable to convert PDF to images. Is Poppler installed/in PATH? Details: {e}"

    text = []
    try:
        for img in images:
            if isinstance(img, Image.Image):
                text.append(pytesseract.image_to_string(img))
            else:
                text.append(pytesseract.image_to_string(Image.fromarray(img)))
        final = clean_text("\n".join(text))
        if not final:
            return "", "OCR ran but extracted empty text."
        return final, "OCR used"
    except Exception as e:
        return "", f"OCR failed during text extraction. Details: {e}"

def call_llm(system_prompt: str, user_prompt: str) -> str:
    if not client:
        return "Missing OPENAI_API_KEY. Add it locally to .env or in Streamlit Cloud Secrets."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

def keywords_from_text(text: str) -> list[str]:
    # simple keyword extraction: keep meaningful words + common tech tokens
    text = text.lower()
    # keep words / numbers / + # .
    tokens = re.findall(r"[a-z0-9\+\#\.]{2,}", text)
    stop = {
        "the","and","for","with","that","this","from","you","your","are","was","were","will","can","able",
        "have","has","had","but","not","all","any","our","their","they","them","his","her","she","him",
        "in","on","at","to","of","as","an","a","is","it","be","or","by","we","i"
    }
    tokens = [t for t in tokens if t not in stop]
    return tokens

def match_score(resume_text: str, job_text: str) -> dict:
    """
    Computes a simple match score based on keyword overlap.
    Returns dict with score, missing_keywords, matched_keywords.
    """
    r = set(keywords_from_text(resume_text))
    j = keywords_from_text(job_text)
    j_counts = Counter(j)

    # choose top-ish job keywords (frequency weighted)
    job_top = [w for w, _ in j_counts.most_common(40)]
    job_set = set(job_top)

    matched = sorted(list(job_set.intersection(r)))
    missing = sorted(list(job_set.difference(r)))

    if not job_set:
        score = 0
    else:
        score = int(round(100 * (len(matched) / len(job_set))))

    return {
        "score": score,
        "matched_keywords": matched[:30],
        "missing_keywords": missing[:30],
        "job_keywords_used": sorted(list(job_set))[:60],
    }


# =========================
# UI
# =========================
st.title("AI Job Application Toolkit")
st.caption("Resume feedback • Job match score • Cover letter generator (PDF + OCR supported)")

if not API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it locally in a .env file or in Streamlit Cloud → App Settings → Secrets.")

col_left, col_right = st.columns([1, 1.05])

with col_left:
    st.header("1) Upload")
    uploaded_file = st.file_uploader("Resume (PDF)", type=["pdf"])

    st.header("2) Settings")
    target_role = st.text_input("Target role", value="AI / Automation Intern (or Entry-Level)")
    tone = st.selectbox(
        "Resume feedback tone",
        ["Strict (ATS-focused)", "Balanced", "Supportive"],
        index=0
    )
    show_preview = st.toggle("Show extracted text preview", value=False)

    st.subheader("Paste Job Description (required for Match Score)")
    job_desc = st.text_area("Job description", height=180, placeholder="Paste the job post here...")

    st.header("3) Cover letter settings")
    company = st.text_input("Company name (optional)", value="")
    hiring_manager = st.text_input("Hiring manager (optional)", value="")
    cl_length = st.selectbox("Cover letter length", ["Short (150–220 words)", "Standard (250–350 words)", "Long (400–550 words)"], index=1)

with col_right:
    st.header("Output")
    out_box = st.empty()

resume_text = ""
extract_note = ""

if uploaded_file:
    pdf_bytes = uploaded_file.read()

    # 1) Try embedded text first
    resume_text = extract_text_pypdf(pdf_bytes)

    # 2) OCR fallback if empty
    if not resume_text:
        resume_text, extract_note = extract_text_ocr(pdf_bytes)

    if extract_note:
        st.info(extract_note)

    if not resume_text:
        st.error("Could not extract text from this PDF.")
    else:
        if show_preview:
            st.subheader("Extracted text preview")
            st.code(resume_text[:4000] + ("\n...\n" if len(resume_text) > 4000 else ""), language="text")

# Buttons row
b1, b2, b3 = st.columns([1, 1, 1])
analyze_clicked = b1.button("Analyze Resume", use_container_width=True, type="primary", disabled=not bool(resume_text))
match_clicked = b2.button("Get Match Score", use_container_width=True, disabled=not (resume_text and job_desc.strip()))
cover_clicked = b3.button("Generate Cover Letter", use_container_width=True, disabled=not (resume_text and job_desc.strip()))

# Results state
if "resume_feedback" not in st.session_state:
    st.session_state["resume_feedback"] = ""
if "match_result" not in st.session_state:
    st.session_state["match_result"] = None
if "cover_letter" not in st.session_state:
    st.session_state["cover_letter"] = ""

# Analyze Resume
if analyze_clicked:
    sys = "You are a professional ATS resume reviewer. Be clear, practical, and structured."
    user = f"""
Target role: {target_role}
Tone: {tone}

Resume text:
{resume_text}

Return EXACTLY this format:

Score: X/10
Top strengths:
- ...
Biggest gaps:
- ...
Fixes that matter most (priority order):
1) ...
ATS keywords to add:
- ...
Professional Summary rewrite:
"..."
"""
    st.session_state["resume_feedback"] = call_llm(sys, user)

# Match Score
if match_clicked:
    st.session_state["match_result"] = match_score(resume_text, job_desc)

# Cover Letter
if cover_clicked:
    length_hint = {
        "Short (150–220 words)": "150–220 words",
        "Standard (250–350 words)": "250–350 words",
        "Long (400–550 words)": "400–550 words",
    }[cl_length]

    sys = "You write concise, tailored, professional cover letters. Avoid fluff. Use a confident but not arrogant tone."
    user = f"""
Write a {length_hint} cover letter tailored to the job description, using the resume.

Company: {company or "(not provided)"}
Hiring manager: {hiring_manager or "(not provided)"}
Target role: {target_role}

Resume:
{resume_text}

Job description:
{job_desc}

Rules:
- 3–5 short paragraphs max
- Mention 2–4 relevant skills/experiences from the resume
- Mirror important keywords from the job description
- End with a simple call to action
"""
    st.session_state["cover_letter"] = call_llm(sys, user)

# Render Output
with col_right:
    if st.session_state["resume_feedback"]:
        st.subheader("Resume feedback")
        st.markdown(st.session_state["resume_feedback"])

        st.download_button(
            "Download resume feedback (.txt)",
            data=st.session_state["resume_feedback"].encode("utf-8"),
            file_name="resume_feedback.txt",
            mime="text/plain",
        )

    if st.session_state["match_result"]:
        st.subheader("Job Match Score")
        m = st.session_state["match_result"]
        st.metric("Match Score", f"{m['score']}%")

        st.write("**Matched keywords (sample):**")
        st.write(", ".join(m["matched_keywords"]) if m["matched_keywords"] else "None found")

        st.write("**Missing keywords to add (sample):**")
        st.write(", ".join(m["missing_keywords"]) if m["missing_keywords"] else "None — great overlap")

        st.download_button(
            "Download match report (.json)",
            data=json.dumps(m, indent=2).encode("utf-8"),
            file_name="match_report.json",
            mime="application/json",
        )

    if st.session_state["cover_letter"]:
        st.subheader("Cover letter")
        st.markdown(st.session_state["cover_letter"])

        st.download_button(
            "Download cover letter (.txt)",
            data=st.session_state["cover_letter"].encode("utf-8"),
            file_name="cover_letter.txt",
            mime="text/plain",
        )

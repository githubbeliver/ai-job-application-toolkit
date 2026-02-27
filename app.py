import os
import re
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

import pytesseract
from pdf2image import convert_from_bytes

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="AI Job Application Toolkit", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      section.main > div { padding-top: 1rem; }
      .muted { opacity: 0.75; }
    </style>
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0.25rem;">AI Job Application Toolkit</h1>
        <p class="muted" style="margin-top: 0;">
            Resume feedback • Job match score • Cover letter generator (PDF + OCR supported)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Environment / Paths
# ----------------------------
load_dotenv()

images = convert_from_bytes(pdf_bytes)
text = ""
for image in images:
    text += pytesseract.image_to_string(image)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY in .env file. Add: OPENAI_API_KEY=your_key")
    st.stop()

client = OpenAI(api_key=api_key)

# ----------------------------
# Helpers
# ----------------------------
def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_pdf_text(pdf_file) -> str:
    try:
        reader = PdfReader(pdf_file)
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        return clean_text("\n".join(parts))
    except Exception:
        return ""

def ocr_pdf(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, poppler_path=POPPLER_BIN)
    parts = [pytesseract.image_to_string(img) for img in images]
    return clean_text("\n".join(parts))

def parse_score_out_of_10(text: str):
    m = re.search(r"(\d+)\s*(?:out of)?\s*/?\s*10", text, re.IGNORECASE)
    if not m:
        return None
    try:
        val = int(m.group(1))
        return max(0, min(10, val))
    except Exception:
        return None

def render_resume_feedback(result: str):
    score = parse_score_out_of_10(result)
    if score is not None:
        st.subheader("Overall Score")
        st.progress(score / 10)
        st.caption(f"{score}/10")
        st.divider()

    parts = re.split(r"\n##\s+", "\n" + result.strip())
    for part in parts:
        part = part.strip()
        if not part:
            continue
        title, *rest = part.split("\n", 1)
        content = rest[0] if rest else ""
        expanded = title.lower().startswith("score")
        with st.expander(title.strip(), expanded=expanded):
            st.markdown(content)

# ----------------------------
# Prompts
# ----------------------------
def build_resume_prompt(resume_text: str, target_role: str, tone: str, job_desc: str) -> str:
    tone_map = {
        "Normal": "Be direct but encouraging.",
        "Strict (ATS-focused)": "Be strict, ATS-optimized, and very specific.",
        "Friendly": "Be supportive and friendly while still giving actionable feedback.",
    }
    style = tone_map.get(tone, tone_map["Normal"])
    jd_part = f"\nJOB DESCRIPTION:\n{job_desc}\n" if job_desc.strip() else ""

    return f"""
You are a professional resume reviewer. {style}

Target role: {target_role}
{jd_part}

Return your response in EXACTLY this structure, using markdown headings:

## Score
Give a score out of 10 and a one-sentence justification.

## Top strengths
List 5 bullet points.

## Biggest gaps
List 5 bullet points.

## Fixes that matter most (priority order)
List 7 bullet points with short examples.

## ATS keywords to add
Provide 10–20 keywords/phrases relevant to the target role (and job description if provided).

## Rewrite: Professional Summary (3–4 lines)
Write a strong summary tailored to the target role (and job description if provided).

## Rewrite: 3 bullets for most recent job
Write 3 impact bullets using numbers/metrics where possible.

RESUME TEXT:
{resume_text}
""".strip()

def analyze_resume(resume_text: str, target_role: str, tone: str, job_desc: str) -> str:
    prompt = build_resume_prompt(resume_text, target_role, tone, job_desc)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You produce structured, high-quality resume feedback."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

def build_cover_letter_prompt(resume_text: str, target_role: str, job_desc: str, letter_tone: str, length: str) -> str:
    tone_map = {
        "Professional": "Professional, confident, and clear.",
        "Warm": "Warm and personable but still professional.",
        "Bold": "Bold, high-energy, and persuasive (still professional).",
    }
    length_map = {
        "Short (150–200 words)": "150–200 words, tight and punchy.",
        "Standard (250–350 words)": "250–350 words, standard cover letter length.",
        "Long (400–500 words)": "400–500 words, more detailed but not repetitive.",
    }

    jd_rules = (
        "- Use the job description to mirror key responsibilities and keywords.\n"
        "- Mention 2–4 specific skills/tools that match the JD.\n"
        if job_desc.strip()
        else "- No job description provided: keep it broadly aligned to the target role.\n"
    )

    return f"""
Write a cover letter for the target role: {target_role}

Tone: {tone_map.get(letter_tone, tone_map["Professional"])}
Length: {length_map.get(length, length_map["Standard (250–350 words)"])}

Rules:
- Use ONLY information supported by the resume text. Do not invent employers, degrees, or projects.
{jd_rules}
- Format as a real cover letter with:
  1) Greeting (use "Dear Hiring Manager," if company unknown)
  2) 2–3 short body paragraphs
  3) Closing + call to action
- Avoid filler like "I believe I am the perfect candidate."
- Make it ATS-friendly and specific.

JOB DESCRIPTION (if any):
{job_desc.strip() if job_desc.strip() else "(none)"}

RESUME TEXT:
{resume_text}
""".strip()

def generate_cover_letter(resume_text: str, target_role: str, job_desc: str, letter_tone: str, length: str) -> str:
    prompt = build_cover_letter_prompt(resume_text, target_role, job_desc, letter_tone, length)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You write excellent cover letters without fabricating details."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content

# --- NEW: Job match scoring (returns JSON we can render) ---
def build_match_prompt(resume_text: str, job_desc: str, target_role: str) -> str:
    return f"""
You are an ATS/job-matching evaluator.

Compare the RESUME to the JOB DESCRIPTION for the target role: {target_role}.

Return ONLY valid JSON with these keys:
- "match_score": integer 0-100 (overall match)
- "top_matches": array of 8-15 keywords/skills the resume already contains that match the JD
- "missing_keywords": array of 10-25 keywords/skills the resume is missing (high impact)
- "quick_fixes": array of 3-6 bullet suggestions (strings) to improve match WITHOUT lying
- "one_sentence_summary": string summarizing fit in one sentence

Rules:
- Do NOT invent experience.
- Prefer concrete skills/tools/phrases from the JD.
- Keep keywords short (1-4 words) when possible.
- JSON only. No markdown.

JOB DESCRIPTION:
{job_desc}

RESUME:
{resume_text}
""".strip()

def get_match_report(resume_text: str, job_desc: str, target_role: str) -> dict:
    prompt = build_match_prompt(resume_text, job_desc, target_role)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return only JSON. No markdown. No extra text."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = resp.choices[0].message.content.strip()

    # In case model wraps JSON in accidental text, try to extract the first JSON object
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("Could not parse JSON match report.")
        return json.loads(m.group(0))

def render_match_report(report: dict):
    score = int(report.get("match_score", 0))
    score = max(0, min(100, score))

    st.subheader("Match Score")
    st.progress(score / 100)
    st.markdown(f"### {score}%")

    summary = report.get("one_sentence_summary", "")
    if summary:
        st.caption(summary)

    st.divider()

    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### ✅ Top matches found")
        top_matches = report.get("top_matches", [])
        if top_matches:
            st.write(top_matches)
        else:
            st.write("(none detected)")

    with c2:
        st.markdown("#### ❗ Missing high-impact keywords")
        missing = report.get("missing_keywords", [])
        if missing:
            st.write(missing)
        else:
            st.write("(none detected)")

    st.markdown("#### 🚀 Quick fixes (without lying)")
    fixes = report.get("quick_fixes", [])
    if fixes:
        for f in fixes:
            st.write(f"- {f}")
    else:
        st.write("(none)")

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("## 1) Upload")
    uploaded_file = st.file_uploader("Resume (PDF)", type="pdf")

    st.markdown("## 2) Settings")
    target_role = st.text_input("Target role", value="AI / Automation Intern (or Entry-Level)")
    tone = st.selectbox("Resume feedback tone", ["Normal", "Strict (ATS-focused)", "Friendly"], index=1)
    show_preview = st.toggle("Show extracted text preview", value=False)

    job_desc = st.text_area("Paste Job Description (required for Match Score)", height=160)

    st.markdown("## 3) Cover letter settings")
    letter_tone = st.selectbox("Cover letter tone", ["Professional", "Warm", "Bold"], index=0)
    letter_length = st.selectbox(
        "Cover letter length",
        ["Short (150–200 words)", "Standard (250–350 words)", "Long (400–500 words)"],
        index=1,
    )

with right:
    st.markdown("## Output")
    if "resume_feedback" not in st.session_state and "cover_letter" not in st.session_state and "match_report" not in st.session_state:
        st.info("Upload a resume to get started.")

# ----------------------------
# Processing
# ----------------------------
resume_text = ""
used_ocr = False

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()

    with st.spinner("Extracting text from PDF..."):
        selectable_text = extract_pdf_text(uploaded_file)

    resume_text = selectable_text

    if not resume_text:
        used_ocr = True
        with st.spinner("No selectable text found — running OCR (may take 10–30 seconds)..."):
            try:
                resume_text = ocr_pdf(pdf_bytes)
            except Exception as e:
                st.error("OCR failed. Check Poppler/Tesseract paths.\n\n" + str(e))
                st.stop()

    resume_text = clean_text(resume_text)

    if not resume_text:
        st.error("Still couldn't extract text. Try exporting as a text-based PDF (not scanned).")
        st.stop()

    st.caption("🧾 OCR used" if used_ocr else "✅ Selectable text")

    if show_preview:
        st.markdown("## Extracted text preview")
        st.code(resume_text[:2000] + ("\n...\n" if len(resume_text) > 2000 else ""))

    b1, b2, b3 = st.columns(3)
    with b1:
        run_resume = st.button("Analyze Resume", type="primary")
    with b2:
        run_match = st.button("Get Match Score", type="secondary", disabled=(not job_desc.strip()))
    with b3:
        run_cover = st.button("Generate Cover Letter", type="secondary")

    if run_resume:
        with st.spinner("Analyzing resume with AI..."):
            st.session_state["resume_feedback"] = analyze_resume(resume_text, target_role, tone, job_desc)

    if run_match:
        with st.spinner("Scoring match vs job description..."):
            st.session_state["match_report"] = get_match_report(resume_text, job_desc, target_role)

    if run_cover:
        with st.spinner("Generating cover letter..."):
            st.session_state["cover_letter"] = generate_cover_letter(
                resume_text, target_role, job_desc, letter_tone, letter_length
            )

# ----------------------------
# Output Tabs
# ----------------------------
tabs = []
if "resume_feedback" in st.session_state:
    tabs.append("📌 Resume Feedback")
if "match_report" in st.session_state:
    tabs.append("🎯 Match Score")
if "cover_letter" in st.session_state:
    tabs.append("✉️ Cover Letter")
if any(k in st.session_state for k in ["resume_feedback", "match_report", "cover_letter"]):
    tabs.append("⬇️ Download")

if tabs:
    t = st.tabs(tabs)
    i = 0

    if "resume_feedback" in st.session_state:
        with t[i]:
            render_resume_feedback(st.session_state["resume_feedback"])
        i += 1

    if "match_report" in st.session_state:
        with t[i]:
            render_match_report(st.session_state["match_report"])
        i += 1

    if "cover_letter" in st.session_state:
        with t[i]:
            st.subheader("Generated cover letter")
            st.markdown(st.session_state["cover_letter"])
        i += 1

    with t[i]:
        st.write("Download your outputs as .txt or .json:")

        if "resume_feedback" in st.session_state:
            st.download_button(
                "Download resume feedback (.txt)",
                st.session_state["resume_feedback"].encode("utf-8"),
                "resume_feedback.txt",
                "text/plain",
            )

        if "cover_letter" in st.session_state:
            st.download_button(
                "Download cover letter (.txt)",
                st.session_state["cover_letter"].encode("utf-8"),
                "cover_letter.txt",
                "text/plain",
            )

        if "match_report" in st.session_state:
            st.download_button(
                "Download match report (.json)",
                json.dumps(st.session_state["match_report"], indent=2).encode("utf-8"),
                "match_report.json",
                "application/json",
            )

import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pdf2image import convert_from_bytes
from pypdf import PdfReader
import pytesseract

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Missing OPENAI_API_KEY in .env file or Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI Job Application Toolkit", layout="wide")

st.title("AI Job Application Toolkit")
st.caption("Resume feedback • Job match score • Cover letter generator (PDF + OCR supported)")

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        # Try normal PDF text extraction first
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if text.strip():
            return text

        # Fallback to OCR
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.read()
        images = convert_from_bytes(pdf_bytes)

        ocr_text = ""
        for image in images:
            ocr_text += pytesseract.image_to_string(image)

        return ocr_text

    except Exception as e:
        st.error(f"OCR failed: {e}")
        return ""

# -----------------------------
# AI Resume Feedback
# -----------------------------
def get_resume_feedback(resume_text, target_role):
    prompt = f"""
    You are an ATS resume analyzer.

    Target Role: {target_role}

    Analyze this resume and provide:
    1) Score out of 10
    2) Top strengths
    3) Biggest gaps
    4) Improvements in priority order
    5) ATS keywords to add

    Resume:
    {resume_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    return response.choices[0].message.content

# -----------------------------
# Match Score
# -----------------------------
def get_match_score(resume_text, job_description):
    prompt = f"""
    Compare the resume to the job description.

    Provide:
    1) Match Score (0–100%)
    2) Missing keywords
    3) Strength alignment summary

    Resume:
    {resume_text}

    Job Description:
    {job_description}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content

# -----------------------------
# Cover Letter Generator
# -----------------------------
def generate_cover_letter(resume_text, job_description, length):
    prompt = f"""
    Write a professional cover letter.

    Length: {length}

    Resume:
    {resume_text}

    Job Description:
    {job_description}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    return response.choices[0].message.content

# -----------------------------
# UI Layout
# -----------------------------

st.header("1) Upload")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

st.header("2) Settings")

target_role = st.text_input("Target Role", "AI / Automation Intern (or Entry-Level)")
tone = st.selectbox("Resume feedback tone", ["Strict (ATS-focused)", "Balanced", "Encouraging"])

job_description = st.text_area("Paste Job Description (required for Match Score)")

st.header("3) Cover Letter Settings")
cover_length = st.selectbox(
    "Cover letter length",
    ["Short (150–200 words)", "Standard (250–350 words)", "Detailed (400+ words)"]
)

# -----------------------------
# Buttons
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    analyze_clicked = st.button("Analyze Resume")

with col2:
    match_clicked = st.button("Get Match Score")

with col3:
    cover_clicked = st.button("Generate Cover Letter")

# -----------------------------
# Processing Logic
# -----------------------------

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.error("Could not extract text from this PDF.")
    else:

        if analyze_clicked:
            with st.spinner("Analyzing resume..."):
                feedback = get_resume_feedback(resume_text, target_role)
                st.subheader("Resume Feedback")
                st.write(feedback)

        if match_clicked:
            if not job_description.strip():
                st.warning("Please paste a job description first.")
            else:
                with st.spinner("Calculating match score..."):
                    match = get_match_score(resume_text, job_description)
                    st.subheader("Job Match Analysis")
                    st.write(match)

        if cover_clicked:
            if not job_description.strip():
                st.warning("Please paste a job description first.")
            else:
                with st.spinner("Generating cover letter..."):
                    letter = generate_cover_letter(resume_text, job_description, cover_length)
                    st.subheader("Generated Cover Letter")
                    st.write(letter)

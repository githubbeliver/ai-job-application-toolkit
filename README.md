# AI Job Application Toolkit 🚀

An AI-powered resume analyzer that provides ATS-friendly feedback, calculates a job match score against a job description, and generates tailored cover letters.

🔗 Live App: https://YOUR-STREAMLIT-LINK.streamlit.app

---

## ✨ Features

- 📄 Resume PDF upload
- 🔎 OCR fallback for scanned PDFs (Tesseract + Poppler)
- 📊 ATS-style scoring system
- 🎯 Job match score (%) vs job description
- 🧠 Missing keyword detection
- ✍️ AI-generated tailored cover letters
- 💾 Download outputs (.txt / .json)

---

## 🛠 Tech Stack

- Python
- Streamlit
- OpenAI API
- Tesseract OCR
- Poppler
- pypdf / pdf2image / Pillow

---

## 🧠 How It Works

1. Upload resume (PDF)
2. Text is extracted (OCR if needed)
3. OpenAI analyzes resume structure + skills
4. If job description is provided:
   - Calculates semantic match score
   - Identifies missing keywords
5. Generates a tailored cover letter

---

## ⚙️ Run Locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

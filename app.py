import streamlit as st
import fitz  # PyMuPDF
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Domain-specific keywords
DOMAIN_KEYWORDS = {
    "Statutory Audit": ["Auditing", "External Audit", "Financial Reporting", "Accounting", "IFRS"],
    "Internal Audit": ["Internal Audit", "Risk Management", "SOX regulations", "Fraud investigation"],
    "Direct Tax": ["Income Tax", "Tax Compliance", "Corporate Tax", "Tax Returns"],
    "Indirect Tax": ["GST", "Tax Compliances", "GST return filing", "GST Audits"],
    "Transfer Pricing": ["Transfer Pricing", "Valuation", "Benchmarking Analysis"],
    "FP&A": ["Financial Planning", "Budgeting", "Forecasting", "Variance Analysis"],
    "M&A": ["Mergers & Acquisitions", "Due Diligence", "Valuation", "Corporate Finance"],
    "Management Consulting": ["Business Strategy", "Change Management", "ERP", "Performance Management"],
    "Credit Manager": ["Credit Risk", "Credit Analysis", "Portfolio Management", "Underwriting"]
}

# Helper to extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Match logic
def get_match_score(resume_text, jd_text, keywords):
    combined_keywords = " ".join(keywords)
    corpus = [resume_text, jd_text + " " + combined_keywords]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 10, 2)

# Missing keywords
def get_missing_keywords(resume_text, keywords):
    missing = [kw for kw in keywords if re.search(rf"\b{re.escape(kw)}\b", resume_text, re.IGNORECASE) is None]
    return missing

# Streamlit UI
st.set_page_config(page_title="Resume to JD Match", layout="centered")
st.title("🔍 Resume to JD Match Tool")

domain = st.selectbox("Select Domain", list(DOMAIN_KEYWORDS.keys()))
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if st.button("Analyze") and resume_file and jd_file:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        jd_text = extract_text_from_pdf(jd_file)
        keywords = DOMAIN_KEYWORDS[domain]
        score = get_match_score(resume_text, jd_text, keywords)
        missing_keywords = get_missing_keywords(resume_text, keywords)

    st.success(f"✅ Match Score: {score}/10")
    if missing_keywords:
        st.markdown("**🔑 Suggested Keywords to Include:**")
        st.write(", ".join(missing_keywords))
    else:
        st.markdown("✅ Your resume contains all relevant keywords!")
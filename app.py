import os
import re
import nltk
import spacy
import pandas as pd
import numpy as np
from docx import Document
from ftfy import fix_text
from pyresparser import ResumeParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# -------------------------
# SETUP
# -------------------------
@st.cache_resource
def setup_nltk():
    data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    os.makedirs(data_dir, exist_ok=True)
    nltk.data.path.append(data_dir)
    nltk.download('stopwords', download_dir=data_dir, quiet=True)
    nltk.download('punkt', download_dir=data_dir, quiet=True)
    nltk.download('wordnet', download_dir=data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=data_dir, quiet=True)
    nltk.download('maxent_ne_chunker', download_dir=data_dir, quiet=True)
    nltk.download('words', download_dir=data_dir, quiet=True)
    return True

setup_nltk()
stopw = set(nltk.corpus.stopwords.words('english'))

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# -------------------------
# LOAD JOB DATA
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('all_jobs.csv')
    df.rename(columns={
        'title': 'Position',
        'company': 'Company',
        'location': 'Location',
        'description': 'Job_Description'
    }, inplace=True)
    df['clean'] = df['Job_Description'].apply(
        lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word.lower() not in stopw])
    )
    return df

df = load_data()

# -------------------------
# STREAMLIT UI
# -------------------------
st.title("💼 AI Resume Parser and Job Recommendation System")
st.write("Upload your resume to get personalized job recommendations!")

uploaded_file = st.file_uploader("📂 Upload your Resume (.docx or .pdf)", type=["docx", "pdf"])

if uploaded_file is not None:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded: {uploaded_file.name}")

    try:
        data = ResumeParser(uploaded_file.name).get_extracted_data()
        resume_skills = data.get('skills', [])
        st.subheader("🧠 Extracted Skills")
        st.write(", ".join(resume_skills))

        if len(resume_skills) == 0:
            st.warning("No skills found in your resume.")
        else:
            # TF-IDF Vectorizer
            skills_text = [' '.join(resume_skills)]
            def ngrams(string, n=3):
                string = fix_text(string)
                string = string.encode("ascii", errors="ignore").decode()
                string = string.lower()
                chars_to_remove = [")", "(", ".", "|", "[", "]", "{", "}", "'"]
                rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
                string = re.sub(rx, '', string)
                string = string.replace('&', 'and')
                string = string.replace(',', ' ')
                string = string.replace('-', ' ')
                string = string.title()
                string = re.sub(' +', ' ', string).strip()
                string = ' ' + string + ' '
                string = re.sub(r'[,-./]|\sBD', r'', string)
                ngrams = zip(*[string[i:] for i in range(n)])
                return [''.join(ngram) for ngram in ngrams]

            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
            tfidf = vectorizer.fit_transform(skills_text)
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
            unique_org = (df['clean'].values)

            def getNearestN(query):
                queryTFIDF_ = vectorizer.transform(query)
                distances, indices = nbrs.kneighbors(queryTFIDF_)
                return distances, indices

            distances, indices = getNearestN(unique_org)
            matches = [round(d[0], 2) for d in distances]
            df['Match_Confidence'] = matches
            df_sorted = df.sort_values('Match_Confidence').head(10)

            st.subheader("🎯 Top Job Recommendations")
            for _, row in df_sorted.iterrows():
                st.markdown(f"""
                **Position:** {row['Position']}  
                **Company:** {row['Company']}  
                **Location:** {row['Location']}  
                [🔗 Job Link]({row['link']})
                ---
                """)

    except Exception as e:
        st.error(f"Error parsing resume: {e}")

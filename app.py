from pyresparser import ResumeParser
from docx import Document
from flask import Flask, render_template, redirect, request
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import re
from ftfy import fix_text
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os
import PyPDF2

# -------------------------
# NLTK SETUP
# -------------------------
try:
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    stopw = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning initializing stopwords: {e}")
    stopw = set()

# -------------------------
# LOAD SCRAPED JOB DATA
# -------------------------
df = pd.read_csv('all_jobs.csv')

df.rename(columns={
    'title': 'Position',
    'company': 'Company',
    'location': 'Location',
    'description': 'Job_Description'
}, inplace=True)

if 'link' not in df.columns:
    df['link'] = ''

df['Location'] = df['Location'].astype(str)
df['Location'] = df['Location'].str.replace(r'[^\x00-\x7F]', '', regex=True)
df['Location'] = df['Location'].str.replace("â€“", "", regex=False)

def clean_text(x):
    return ' '.join(
        word for word in str(x).split()
        if len(word) > 2 and word.lower() not in stopw
    )

df['test'] = df['Job_Description'].apply(clean_text)
print(df["Location"])

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

UPLOAD_FOLDER = 'Uploaded_Resumes'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route("/home")
def home():
    return redirect('/')


@app.route('/submit', methods=['GET', 'POST'])
def submit_data():
    if request.method == 'POST':
        try:
            f = request.files['userfile']
            filename = secure_filename(f.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)
            print("Saved file:", filepath)

            lower_name = filename.lower()
            skills_text = ""

            # ------------ PDF RESUMES ------------
            if lower_name.endswith('.pdf'):
                try:
                    with open(filepath, 'rb') as pdf_file:
                        reader = PyPDF2.PdfReader(pdf_file)
                        text_pages = []
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            text_pages.append(page_text)
                        skills_text = "\n".join(text_pages).strip()
                    print("Extracted PDF text length:", len(skills_text))
                except Exception as e:
                    print("PDF read failed:", e)

            # ------------ DOCX RESUMES ------------
            elif lower_name.endswith('.docx'):
                try:
                    doc = Document(filepath)
                    full_text = "\n".join(p.text for p in doc.paragraphs)
                    skills_text = full_text.strip()
                    print("Extracted DOCX text length:", len(skills_text))
                except Exception as e:
                    print("DOCX read failed:", e)

            # ------------ OTHER FORMATS (fallback to ResumeParser) ------------
            if not skills_text:
                try:
                    from pyresparser import ResumeParser
                    data = ResumeParser(filepath).get_extracted_data()
                    print("Parsed data from ResumeParser:", data)

                    if isinstance(data, dict):
                        parts = []
                        for v in data.values():
                            if isinstance(v, list):
                                parts.append(" ".join(str(x) for x in v))
                            else:
                                parts.append(str(v))
                        skills_text = " ".join(parts).strip()
                except Exception as e:
                    print("ResumeParser failed:", e)

            # If still nothing, give error
            if not skills_text or not skills_text.strip():
                return "Could not extract meaningful text from your resume. Please upload a text-based PDF or DOCX.", 400

            # ------------ USE RESUME TEXT FOR MATCHING ------------
            skills = [skills_text]
            org_name_clean = skills

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
                ngrams_list = zip(*[string[i:] for i in range(n)])
                return [''.join(ngram) for ngram in ngrams_list]

            if not any(org_name_clean):
                return "Empty resume text after preprocessing.", 400

            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
            tfidf = vectorizer.fit_transform(org_name_clean)
            print('Vectorizing completed...')

            def getNearestN(query):
                queryTFIDF_ = vectorizer.transform(query)
                distances, indices = nbrs.kneighbors(queryTFIDF_)
                return distances, indices

            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
            unique_org = df['test'].values
            distances, indices = getNearestN(unique_org)

            matches = []
            for i, j in enumerate(indices):
                dist = round(distances[i][0], 2)
                matches.append([dist])

            matches = pd.DataFrame(matches, columns=['Match confidence'])
            df['match'] = matches['Match confidence']

            df1 = df.sort_values('match')
            df2 = df1[['Position', 'Company', 'Location', 'link']].head(10).reset_index()

            dropdown_locations = sorted(df2['Location'].unique())

            job_list = []
            for _, row in df2.iterrows():
                job_list.append({
                    'Position': row['Position'],
                    'Company': row['Company'],
                    'Location': row['Location'],
                    'Link': row['link']
                })

            return render_template('results.html', job_list=job_list, dropdown_locations=dropdown_locations)

        except Exception as e:
            print("Error in /submit:", e)
            return f"An error occurred while processing your resume: {e}", 500

    return redirect('/')



@app.route('/filter', methods=['GET'])
def filter_jobs():
    location = request.args.get('location')
    filtered = df.copy()
    if location:
        filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]
    df2 = filtered[['Position', 'Company', 'Location', 'link']].head(10).reset_index(drop=True)

    # Build job_list with same keys as submit_data (use 'Link' capitalized)
    job_list = []
    for _, row in df2.iterrows():
        job_list.append({
            'Position': row['Position'],
            'Company': row['Company'],
            'Location': row['Location'],
            'Link': row['link']
        })

    dropdown_locations = sorted(df['Location'].unique())
    return render_template('results.html', job_list=job_list, dropdown_locations=dropdown_locations)


if __name__ == "__main__":
    app.run(debug=True)

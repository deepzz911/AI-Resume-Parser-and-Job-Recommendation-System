from pyresparser import ResumeParser
from docx import Document
from flask import Flask, render_template, redirect, request
import numpy as np
import pandas as pd
import re
from ftfy import fix_text
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import spacy
import nltk

# -------------------------
# SETUP
# -------------------------
nltk.data.path.append('C:\\Users\\Dell/nltk_data')
nltk.download('stopwords')

stopw = set(stopwords.words('english'))

# -------------------------
# LOAD SCRAPED JOB DATA
# -------------------------
df = pd.read_csv('all_jobs.csv')

# Rename columns to match Flask logic
df.rename(columns={
    'title': 'Position',
    'company': 'Company',
    'location': 'Location',
    'description': 'Job_Description'
}, inplace=True)

# Clean & preprocess job descriptions
df['test'] = df['Job_Description'].apply(
    lambda x: ' '.join([word for word in str(x).split() if len(word) > 2 and word.lower() not in stopw])
)
print(df["Location"])

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

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
            f.save(f.filename)
            print("Saved file:", f.filename)

            # Try reading .docx file and extracting text
            try:
                doc = Document(f.filename)
                print("Document opened successfully")
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                # Load SpaCy model
                nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
                data = ResumeParser(f.filename, custom_nlp=nlp).get_extracted_data()
            
            except Exception as e:
                print("Error opening document:", e)
                data = ResumeParser(f.filename).get_extracted_data()

            # Extract skills from resume
            resume = data.get('skills', [])
            print("Extracted skills:", resume)

            skills = [' '.join(resume)]
            org_name_clean = skills

            # N-gram vectorizer setup
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
            tfidf = vectorizer.fit_transform(org_name_clean)
            print('Vectorizing completed...')

            # Nearest Neighbor Matching
            def getNearestN(query):
                queryTFIDF_ = vectorizer.transform(query)
                distances, indices = nbrs.kneighbors(queryTFIDF_)
                return distances, indices

            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
            unique_org = (df['test'].values)
            distances, indices = getNearestN(unique_org)

            unique_org = list(unique_org)
            matches = []
            for i, j in enumerate(indices):
                dist = round(distances[i][0], 2)
                matches.append([dist])

            matches = pd.DataFrame(matches, columns=['Match confidence'])
            df['match'] = matches['Match confidence']

            # Sort by best match
            df1 = df.sort_values('match')
            df2 = df1[['Position', 'Company', 'Location', 'link']].head(10).reset_index()

            # Clean up location text
            df2['Location'] = df2['Location'].astype(str)
            df2['Location'] = df2['Location'].str.replace(r'[^\x00-\x7F]', '', regex=True)
            df2['Location'] = df2['Location'].str.replace("â€“", "", regex=False)

            dropdown_locations = sorted(df2['Location'].unique())

            # Prepare list of jobs for rendering
            job_list = []
            for _, row in df2.iterrows():
                job_list.append({
                    'Position': row['Position'],
                    'Company': row['Company'],
                    'Location': row['Location'],
                    'Link': row['link']
                })

            # ✅ Return the results page (must return something)
            return render_template('results.html', job_list=job_list, dropdown_locations=dropdown_locations)

        except Exception as e:
            print("Error in /submit:", e)
            return f"An error occurred while processing your resume: {e}", 500

    # ✅ For GET requests, redirect to home
    return redirect('/')

@app.route('/filter', methods=['GET'])
def filter_jobs():
    location = request.args.get('location')
    filtered = df.copy()
    if location:
        filtered = filtered[filtered['Location'].str.contains(location, case=False, na=False)]
    df2 = filtered[['Position', 'Company', 'Location', 'link']].head(10).reset_index(drop=True)
    job_list = df2.to_dict('records')
    dropdown_locations = sorted(df['Location'].unique())
    return render_template('results.html', job_list=job_list, dropdown_locations=dropdown_locations)


# -------------------------
# RUN APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)

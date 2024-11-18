import os
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Create a Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to clean the text
def clean_resume(resume_text):
    # List of words to remove
    useless_words = set([
        # Prepositions
        "about", "above", "across", "after", "against", "along", "among", "around", "as", "at",
        "before", "behind", "below", "beneath", "beside", "besides", "between", "beyond", "by",
        "concerning", "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
        "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
        "since", "through", "throughout", "to", "toward", "under", "underneath", "until", "unto",
        "up", "upon", "with", "within", "without",
        
        # Filler/Stop words
        "a", "an", "the", "and", "or", "but", "if", "because", "while", "when", "where", "how",
        "so", "then", "than", "that", "which", "who", "whom", "whose", "what", "why", "does",
        "do", "doing", "did", "is", "was", "were", "am", "are", "be", "been", "being", "have",
        "has", "had", "having", "will", "shall", "should", "can", "could", "would", "may", 
        "might", "must",
        
        # Common resume/useless words
        "professional", "work", "responsibility", "project", "skills",
        "detail", "ability", "capability", "dedicated", "self-motivated", 
        "proven", "successfully", "goal-oriented", "efficient", "highly", "excellent", "strong",
        "ability", "communication", "interpersonal", "collaborative", "organized", 
        "detail-oriented", "dynamic",
        
        # Single-letter words
        *list("abcdefghijklmnopqrstuvwxyz")
    ])
    #update , reinstated words like team,experience,enthusiastic
    # Normalize case
    resume_text = resume_text.lower()
    
    # Remove URLs
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    
    # Remove hashtags, mentions, and special characters
    resume_text = re.sub(r'#\S+', '', resume_text)  # Remove hashtags
    resume_text = re.sub(r'@\S+', ' ', resume_text)  # Remove mentions
    resume_text = re.sub(r'[{}]+'.format(re.escape(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', resume_text)
    
    # Remove non-ASCII characters
    resume_text = re.sub(r'[^\x00-\x7f]', r' ', resume_text)
    
    # Tokenize and remove stopwords
    words = resume_text.split()
    cleaned_words = [word for word in words if word not in useless_words and len(word) > 1]
    
    # Join cleaned words
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text


# Function to calculate similarity scores
def calculate_similarityCOSINE(job_desc, resumes, model):
    # Encode job description
    job_desc_embedding = model.encode([job_desc])
    resume_embeddings = model.encode(resumes)
    print(type(resume_embeddings))
    # Compute cosine similarity
    similarities = cosine_similarity(job_desc_embedding, resume_embeddings)
    print("====================================================================================\n")
    print(similarities)
    return similarities[0]  # Return similarity scores for the job description

def calculate_similarityQUALITY(resumes):
    """Calculate resume quality scores based on structure and content"""
    professional_terms = {
        'achieved', 'implemented', 'developed', 'managed', 'led', 'coordinated',
        'strategic', 'initiative', 'successful', 'improved', 'increased', 'decreased',
        'portfolio', 'responsibility', 'leadership', 'expertise', 'proficient'
    }
    
    quality_scores = []
    for resume in resumes:
        # Structure analysis
        bullet_points = len(re.findall(r'[•\-\*]', resume))
        sections = len(re.findall(r'\b[A-Z]{2,}.*\n', resume))
        structure_score = min(1.0, (bullet_points/20 + sections/8) / 2)
        
        # Professional language analysis
        words = resume.lower().split()
        prof_terms = sum(1 for word in words if word in professional_terms)
        prof_score = min(1.0, prof_terms / 15)
        
        # Quantifiable achievements
        quantities = len(re.findall(r'\d+%|\$\d+|\d+ years?', resume))
        quant_score = min(1.0, quantities / 10)
        
        # Combined quality score
        quality_score = (structure_score + prof_score + quant_score) / 3
        quality_scores.append(quality_score)
    
    return quality_scores

def calculate_similarityINTERSECTION(job_desc, resumes, model):
    # We won't use the model parameter but keep it for interface compatibility
    
    # Convert job description to set of words
    job_desc_words = set(job_desc.split())
    
    # Calculate overlap scores for each resume
    scores = []
    max_score = 0
    
    # First pass: calculate raw scores and find maximum
    for resume in resumes:
        resume_words = set(resume.split())
        # Count unique words that appear in both texts
        matching_words = len(job_desc_words.intersection(resume_words))
        scores.append(matching_words)
        max_score = max(max_score, matching_words)
    
    # Second pass: normalize scores between 0 and 1
    if max_score > 0:  # Avoid division by zero
        normalized_scores = [score / max_score for score in scores]
    else:
        normalized_scores = [0.0] * len(scores)
    
    return normalized_scores

def calculate_similarity(job_desc, resumes, model, cosine_weight=0.3):
    cosine_scores=calculate_similarityCOSINE(job_desc,resumes,model);
    normalized_overlap_scores = calculate_similarityINTERSECTION(job_desc,resumes,model);
    
    # Calculate weighted combined scores
    word_overlap_weight = 1 - cosine_weight
    combined_scores = []
    
    for cosine_score, overlap_score in zip(cosine_scores, normalized_overlap_scores):
        weighted_score = (cosine_score * cosine_weight) + (overlap_score * word_overlap_weight)
        combined_scores.append(weighted_score)
    
    return combined_scores


# Route for uploading files and processing them
@app.route('/', methods=['GET', 'POST'])
def upload_files():
    if request.method == 'POST':
        # Save uploaded files
        job_description = request.files.get('job_description')
        resumes = request.files.getlist('resumes')
        

        if not job_description or not resumes:
            return render_template('upload.html', error="Please upload both a job description and resumes.")
        
        # Save job description
        job_desc_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(job_description.filename))
        job_description.save(job_desc_path)
        print(job_desc_path)
        
        # Save resumes
        resume_paths = []
        for resume in resumes:
            resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume.filename))
            resume.save(resume_path)
            resume_paths.append(resume_path)
        print(resume_paths)
        # Process job description
        job_desc_text = clean_resume(extract_text_from_pdf(job_desc_path))
        print("====================================================================================\n")
        print(job_desc_text)
        # Process resumes
        resume_texts = [clean_resume(extract_text_from_pdf(r)) for r in resume_paths]
        resume_names = [os.path.basename(r) for r in resume_paths]

        # Load sentence transformer model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Calculate similarities
        similarities = calculate_similarity(job_desc_text, resume_texts, model)
        print("====================================================================================\n")
        print(similarities)

        # Rank resumes based on similarity
        ranked_resumes = sorted(zip(resume_names, similarities), key=lambda x: x[1], reverse=True)

        # Render results
        return render_template('results.html', ranked_resumes=ranked_resumes)

    return render_template('upload.html')


# # Template for uploading files
# @app.route('/upload', methods=['GET'])
# def upload_template():
#     return '''
#     <!doctype html>
#     <title>Resume Screening</title>
#     <h1>Upload Job Description and Resumes</h1>
#     <form method=post enctype=multipart/form-data>
#       <label for="job_description">Job Description (PDF):</label>
#       <input type=file name=job_description><br><br>
#       <label for="resumes">Resumes (PDFs):</label>
#       <input type=file name=resumes multiple><br><br>
#       <input type=submit value=Upload>
#     </form>
#     '''

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

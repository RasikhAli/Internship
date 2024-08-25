from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import re
from langdetect import detect
from googletrans import Translator

app = Flask(__name__)

# Load preprocessed data
hadith_df_en = pd.read_csv('data/cleaned_hadith_data_english.csv')
# hadith_df_ar = pd.read_csv('data/cleaned_hadith_data_arabic.csv')

embeddings_en = np.load('data/hadith_embeddings_en.npy')
# embeddings_ar = np.load('data/hadith_embeddings_ar.npy')

# Load models for each language
model_en = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# model_ar = SentenceTransformer('bert-base-multilingual-cased')

# Load FAISS indexes for each language
index_en = faiss.read_index('data/hadith_faiss_en.index')
# index_ar = faiss.read_index('data/hadith_faiss_ar.index')

# Initialize the translator
translator = Translator()

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    text = text.lower()  # Convert to lowercase
    return text

# Function to detect the language
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

# Function to retrieve similar Hadiths
def retrieve_similar_hadiths(query, model, index, hadith_df, hadith_type, k=5):
    # Preprocess and embed the query
    query_embedding = model.encode([clean_text(query)])
    
    # Search the FAISS index for similar Hadiths
    distances, indices = index.search(query_embedding, k)
    
    # Initialize an empty list to store the results
    results = []

    # Retrieve and store the top-k Hadiths along with additional information
    for i in range(k):
        index_pos = indices[0][i]
        if(hadith_type == 'en'):
            hadith_text = hadith_df['Hadith'].iloc[index_pos] if not pd.isna(hadith_df['Hadith'].iloc[index_pos]) else 'N/A'
            hadith_grade = hadith_df['English_Grade'].iloc[index_pos] if not pd.isna(hadith_df['English_Grade'].iloc[index_pos]) else 'N/A'
        elif(hadith_type == 'ar'):
            hadith_text = hadith_df['Arabic_Hadith'].iloc[index_pos] if not pd.isna(hadith_df['Arabic_Hadith'].iloc[index_pos]) else 'N/A'
            hadith_grade = hadith_df['Arabic_Grade'].iloc[index_pos] if not pd.isna(hadith_df['Arabic_Grade'].iloc[index_pos]) else 'N/A'
        
        hadith_info = {
            'book_name': hadith_df['Book_Name'].iloc[index_pos] if not pd.isna(hadith_df['Book_Name'].iloc[index_pos]) else 'N/A',
            'chapter_number': hadith_df['Chapter_Number'].iloc[index_pos] if not pd.isna(hadith_df['Chapter_Number'].iloc[index_pos]) else 'N/A',
            'section_number': hadith_df['Section_Number'].iloc[index_pos] if not pd.isna(hadith_df['Section_Number'].iloc[index_pos]) else 'N/A',
            'hadith_number': hadith_df['Hadith_number'].iloc[index_pos] if not pd.isna(hadith_df['Hadith_number'].iloc[index_pos]) else 'N/A',
            'hadith': hadith_text,
            'grade': hadith_grade,
            'distance': float(distances[0][i])
        }
        results.append(hadith_info)

    # If no results were found, return a special message
    if not results:
        return [{'hadith': "JazakAllah for searching, couldn't find matching Hadith"}]

    return results

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling queries
@app.route('/get_similar_hadiths', methods=['POST'])
def get_similar_hadiths():
    user_input = request.form['query']
    lang = detect_language(user_input)

    if lang == 'ar':
        # Translate Arabic query to English
        translated_query = translator.translate(user_input, src='ar', dest='en').text
        # Use the English model and index to retrieve similar Hadiths
        results = retrieve_similar_hadiths(translated_query, model_en, index_en, hadith_df_en, 'ar')
    elif lang == 'ur':
        # Translate Urdu query to English
        translated_query = translator.translate(user_input, src='ur', dest='en').text
        # Use the English model and index to retrieve similar Hadiths
        results = retrieve_similar_hadiths(translated_query, model_en, index_en, hadith_df_en, 'en')
        # Translate the results back to Urdu
        for result in results:
            result['hadith'] = translator.translate(result['hadith'], src='en', dest='ur').text
    else:
        # Default behavior for English and other languages
        results = retrieve_similar_hadiths(user_input, model_en, index_en, hadith_df_en, 'en')

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

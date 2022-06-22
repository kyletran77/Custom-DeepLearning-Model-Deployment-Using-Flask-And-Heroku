
# import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from operator import index
from pandas._config.config import options
import pandas as pd
import fileReader
import Similar
import time

# coding=utf-8

# SQLite for information
import sqlite3
import os

#For image recognition
import json
import pandas as pd
import numpy as np
import spacy
import textract


from collections import Counter
from os import listdir
from os.path import isfile, join

# Flask utils
from flask import Flask, url_for, render_template, request,send_from_directory,redirect
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
app.debug = True

# # Load spacy model

common = {
    'first_name': 'Peter',
    'last_name': 'Parker',
    'alias': 'spiderman'
}

# Reading the CSV files prepared by the fileReader.py
Resumes = pd.read_csv('Resume_Data.csv')
Jobs = pd.read_csv('Job_Data.csv')


############################### JOB DESCRIPTION CODE ######################################

# Asking to Print the Job Desciption Names
if len(Jobs['Name']) > 1:
    index = [a for a in range(len(Jobs['Name']))]


#################################### SCORE CALCUATION ################################
# @st.cache()
def calculate_scores(resumes, job_description, fileName):
    scores = []
    uploaded = resumes.loc[resumes['Name'] == fileName]
    print(type(uploaded['TF_Based']))
    print(type(job_description['TF_Based'][1]))
    for x in range(job_description.shape[0]):
        score = Similar.match(
            uploaded['TF_Based'].to_string(), job_description['TF_Based'][x])
        scores.append(score)
    return scores

############################################ TF-IDF Code ###################################


# @st.cache()
def get_list_of_words(document):
    Document = []

    for a in document:
        raw = a.split(" ")
        Document.append(raw)

    return Document


document = get_list_of_words(Resumes['Cleaned'])

id2word = corpora.Dictionary(document)
corpus = [id2word.doc2bow(text) for text in document]


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=6, random_state=100,
                                            update_every=3, chunksize=100, passes=50, alpha='auto', per_word_topics=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('home.html', common = common)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        img_path = os.path.join(basepath, 'Data/Resumes/', secure_filename(f.filename))
        f.save(img_path)
        f.save(secure_filename(f.filename))
        fileReader.runFileReader()
        # file = open('fileReader.py', 'r').read()
        # exec(file)

        Jobs['Scores'] = calculate_scores(Resumes, Jobs, secure_filename(f.filename))

        Ranked_resumes = Jobs.sort_values(
            by=['Scores'], ascending=False).reset_index(drop=True)

        # Ranked_resumes['Rank'] = pd.DataFrame([i for i in range(1, len(Ranked_resumes['Scores'])+1)])


        
        return render_template('result.html', Pathogen=Ranked_resumes)
        # return result

@app.route('/predict/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', common=common), 404


def get_static_file(path):
    site_root = os.path.realpath(os.path.dirname(__file__))
    return os.path.join(site_root, path)


def get_static_json(path):
    return json.load(open(get_static_file(path)))


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
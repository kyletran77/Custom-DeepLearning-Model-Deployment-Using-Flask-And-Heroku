# # coding=utf-8

# # SQLite for information
# import sqlite3
# import os
# #For image recognition
# import json
# import pandas as pd
# import numpy as np
# import spacy
# import textract


# from spacy.matcher import PhraseMatcher
# from collections import Counter
# from os import listdir
# from os.path import isfile, join

# # Flask utils
# from flask import Flask, url_for, render_template, request,send_from_directory,redirect
# from werkzeug.utils import secure_filename

# # Define a flask app
# app = Flask(__name__)
# app.debug = True

# # Load spacy model
# nlp = spacy.load("en_core_web_sm")


# def create_profile(filename, search_criteria):
    
#     content = textract.process(filename, encoding="utf-8").decode()
#     content = " ".join(content.split()).lower()

#     matcher = PhraseMatcher(nlp.vocab)
#     for col in search_criteria.keys():
#         words = [nlp(word.lower()) for word in search_criteria[col]]
#         matcher.add(col, None, *words)

#     doc = nlp(content)

#     from collections import Counter
#     #value of the current dictionary key
#     d = []
#     matches = matcher(doc)
#     for match_id, start, end in matches:
        
#         rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
#         span = doc[start : end]  # get the matched slice of the doc
#         #tuple main key + text in resume/sub field
#         d.append((rule_id, span.text))
#     print(d)
#     #instead counter-unique- change to frequency
#     a = pd.DataFrame.from_dict(Counter(d), orient='index').reset_index()
#     df = pd.DataFrame(a['index'].tolist(), columns=["Subject", "Keywords"])
#     if len(df)==0.5:
#         df["Count"]=0
#     else:
#         df["Count"] = a[0]
#     df['Candidate'] = filename.split("/")[-1]
    
#     return df

# #redundant
# def rank(path, search_criteria):
#         #iterate through the different resumes
#         # files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
#         final_database = pd.DataFrame()
#         data = create_profile(path, search_criteria)
#         #df being printed out
#         final_database = final_database.append(data, ignore_index=True)
#         #ordering for ranking
#         df = final_database.groupby(['Candidate','Subject']).count().reset_index()
#         print(df.head())
#         #removing null columns
#         df = df.pivot_table(values='Count',index='Candidate', columns='Subject').fillna(0).reset_index()
#         #adding the values of horizontal (aka all the values)
#         df['TOTAL SCORE'] = df.sum(axis=1)
#         df = df.astype("int32", errors="ignore")

#         #cosmetics
#         df['RANKING']=np.round(df['TOTAL SCORE'].rank(pct=True),1)
#         df['RATING'] = df['RANKING'].apply(lambda x:
#             '⭐⭐⭐⭐⭐' if x >= .9 else (
#             '⭐⭐⭐⭐' if x >= .7 else (
#             '⭐⭐⭐' if x >= .5 else (
#             '⭐⭐' if x >= .3 else (
#             '⭐' if x >=.1 else '')))))
#         df = df.sort_values(by='RANKING', ascending=False)

#         return df
    
    

# class ResumeAnalyzer:
    
#     def __init__(self):
        
#         pass
  
#     def rank(self, path, metadata):
#         #get dictionary of requirements
#         self.search_criteria = metadata
#         self.path = path
#         #iterate through the different resumes
#         # files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
#         final_database = pd.DataFrame()

#         # for file_name in files:
#         #     try:
#         #         data = create_profile(os.path.join(self.path,file_name), self.search_criteria)
#         #     except:
#         #         data="XXXX"
#         #         continue
#         data = create_profile(self.path, self.search_criteria)

#         #df being printed out
#         final_database = final_database.append(data, ignore_index=True)
#         #ordering for ranking
#         df = final_database.groupby(['Candidate','Subject']).count().reset_index()
#         print(df.head())
#         #removing null columns
#         df = df.pivot_table(values='Count',index='Candidate', columns='Subject').fillna(0).reset_index()
#         #adding the values of horizontal (aka all the values)
#         df['TOTAL SCORE'] = df.sum(axis=1)
#         df = df.astype("int32", errors="ignore")

#         #cosmetics
#         df['RANKING']=np.round(df['TOTAL SCORE'].rank(pct=True),1)
#         df['RATING'] = df['RANKING'].apply(lambda x:
#             '⭐⭐⭐' if x >= .8 else (
#             '⭐⭐' if x >= .5 else (
#             '⭐' if x >=.2 else '')))
#         df = df.sort_values(by='RANKING', ascending=False)

#         return df

#     def render(self, path, metadata, mode="browser"):
        
#         self.path = path
#         self.search_criteria=metadata
#         self.mode = mode
#         print(rank(self.path, self.search_criteria))
#         # run_dash(self.path, self.search_criteria, self.mode)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     # Main page
#     return render_template('index.html')


# @app.route('/predict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['image']
#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         img_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
#         f.save(img_path)
#         f.save(secure_filename(f.filename))
#         analyzer = ResumeAnalyzer()
#         search_criteria = {
#             #experiences, education, projects, skills
#             "Experiences":["neural networks", "cnn", "rnn", "ann", "lstm", "bert", "transformers", "algorithms"],
#             "Education":["regression", "classification", "clustering", "time series", "summarization", "nlp"],
#             "Projects/Research": ["Emotional","sarimax", "prophet", "holt winters"],
#             "Skills": ["arima","sarimax", "techniques", "holt winters"],
            
#         }
#         df = rank(img_path, search_criteria)
        
#         return render_template('result.html', Pathogen=df)
#         # return result

# @app.route('/predict/<filename>')
# def send_file(filename):
#     return send_from_directory('uploads', filename)



# if __name__ == '__main__':
#     app.run()

# import matplotlib.colors as mcolors
import gensim
import gensim.corpora as corpora
from operator import index
# from wordcloud import WordCloud
from pandas._config.config import options
import pandas as pd
import fileReader
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
import Similar
# from PIL import Image
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


from spacy.matcher import PhraseMatcher
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

# image = Image.open('Images//logo.png')
# st.image(image, use_column_width=True)

# st.title("Resume Matcher")


# Reading the CSV files prepared by the fileReader.py
Resumes = pd.read_csv('Resume_Data.csv')
Jobs = pd.read_csv('Job_Data.csv')


############################### JOB DESCRIPTION CODE ######################################

# Asking to Print the Job Desciption Names
if len(Jobs['Name']) > 1:
    index = [a for a in range(len(Jobs['Name']))]
    # fig = go.Figure(data=[go.Table(header=dict(values=["Job No.", "Job Desc. Name"], line_color='darkslategray',
    #                                             fill_color='lightskyblue'),
    #                                 cells=dict(values=[index, Jobs['Name']], line_color='darkslategray',
    #                                             fill_color='cyan'))
    #                         ])
    # fig.update_layout(width=700, height=400)



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

################################### LDA CODE ##############################################


# @st.cache  # Trying to improve performance by reducing the rerun computations
# def format_topics_sentences(ldamodel, corpus):
#     sent_topics_df = []
#     for i, row_list in enumerate(ldamodel[corpus]):
#         row = row_list[0] if ldamodel.per_word_topics else row_list
#         row = sorted(row, key=lambda x: (x[1]), reverse=True)
#         for j, (topic_num, prop_topic) in enumerate(row):
#             if j == 0:
#                 wp = ldamodel.show_topic(topic_num)
#                 topic_keywords = ", ".join([word for word, prop in wp])
#                 sent_topics_df.append(
#                     [i, int(topic_num), round(prop_topic, 4)*100, topic_keywords])
#             else:
#                 break
#     return sent_topics_df



@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')


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



if __name__ == '__main__':
    app.run()
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
# ALLOWED_EXTENSIONS = set(['pdf')


# # load json file before weights
# loaded_json = open("models/crop.json", "r")
# # read json architecture into variable
# loaded_json_read = loaded_json.read()
# # close file
# loaded_json.close()
# # retreive model from json
# loaded_model = model_from_json(loaded_json_read)
# # load weights
# loaded_model.load_weights("models/crop_weights.h5")
# model1 = load_model("models/one-class.h5")
# global graph
# graph = tf.get_default_graph()


# def info():
#     conn = sqlite3.connect("models/crop.sqlite")
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM crop")
#     rows = cursor.fetchall()
#     return rows


# def leaf_predict(img_path):
#     # load image with target size
#     img = image.load_img(img_path, target_size=(256, 256))
#     # convert to array
#     img = image.img_to_array(img)
#     # normalize the array
#     img /= 255
#     # expand dimensions for keras convention
#     img = np.expand_dims(img, axis=0)

#     with graph.as_default():
#         opt = keras.optimizers.Adam(lr=0.001)
#         loaded_model.compile(optimizer=opt, loss='mse')
#         preds = model1.predict(img)
#         dist = np.linalg.norm(img - preds)
#         if dist <= 20:
#             return "leaf"
#         else:
#             return "not leaf"


# def model_predict(img_path):
#     # load image with target size
#     img = image.load_img(img_path, target_size=(256, 256))
#     # convert to array
#     img = image.img_to_array(img)
#     # normalize the array
#     img /= 255
#     # expand dimensions for keras convention
#     img = np.expand_dims(img, axis=0)

#     with graph.as_default():
#         opt = keras.optimizers.Adam(lr=0.001)
#         loaded_model.compile(
#             optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#         preds = loaded_model.predict_classes(img)
#         return int(preds)


# Load spacy model
nlp = spacy.load("en_core_web_sm")


def create_profile(filename, search_criteria):
    
    content = textract.process(filename, encoding="utf-8").decode()
    content = " ".join(content.split()).lower()

    matcher = PhraseMatcher(nlp.vocab)
    for col in search_criteria.keys():
        words = [nlp(word.lower()) for word in search_criteria[col]]
        matcher.add(col, None, *words)

    doc = nlp(content)

    from collections import Counter
    #value of the current dictionary key
    d = []
    matches = matcher(doc)
    for match_id, start, end in matches:
        
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        #tuple main key + text in resume/sub field
        d.append((rule_id, span.text))
    print(d)
    #instead counter-unique- change to frequency
    a = pd.DataFrame.from_dict(Counter(d), orient='index').reset_index()
    df = pd.DataFrame(a['index'].tolist(), columns=["Subject", "Keywords"])
    if len(df)==0:
        df["Count"]=0
    else:
        df["Count"] = a[0]
    df['Candidate'] = filename.split("/")[-1]
    
    return df

#redundant
def rank(path, search_criteria):
        #iterate through the different resumes
        # files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        final_database = pd.DataFrame()

        # for file_name in files:
        #     try:
        #         data = create_profile(os.path.join(self.path,file_name), self.search_criteria)
        #     except:
        #         data="XXXX"
        #         continue
        data = create_profile(path, search_criteria)

        #df being printed out
        final_database = final_database.append(data, ignore_index=True)
        #ordering for ranking
        df = final_database.groupby(['Candidate','Subject']).count().reset_index()
        print(df.head())
        #removing null columns
        df = df.pivot_table(values='Count',index='Candidate', columns='Subject').fillna(0).reset_index()
        #adding the values of horizontal (aka all the values)
        df['TOTAL SCORE'] = df.sum(axis=1)
        df = df.astype("int32", errors="ignore")

        #cosmetics
        df['RANKING']=np.round(df['TOTAL SCORE'].rank(pct=True),1)
        df['RATING'] = df['RANKING'].apply(lambda x:
            '⭐⭐⭐' if x >= .8 else (
            '⭐⭐' if x >= .5 else (
            '⭐' if x >=.2 else '')))
        df = df.sort_values(by='RANKING', ascending=False)

        return df
    
    

class ResumeAnalyzer:
    
    def __init__(self):
        
        pass
  
    def rank(self, path, metadata):
        #get dictionary of requirements
        self.search_criteria = metadata
        self.path = path
        #iterate through the different resumes
        # files = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        final_database = pd.DataFrame()

        # for file_name in files:
        #     try:
        #         data = create_profile(os.path.join(self.path,file_name), self.search_criteria)
        #     except:
        #         data="XXXX"
        #         continue
        data = create_profile(self.path, self.search_criteria)

        #df being printed out
        final_database = final_database.append(data, ignore_index=True)
        #ordering for ranking
        df = final_database.groupby(['Candidate','Subject']).count().reset_index()
        print(df.head())
        #removing null columns
        df = df.pivot_table(values='Count',index='Candidate', columns='Subject').fillna(0).reset_index()
        #adding the values of horizontal (aka all the values)
        df['TOTAL SCORE'] = df.sum(axis=1)
        df = df.astype("int32", errors="ignore")

        #cosmetics
        df['RANKING']=np.round(df['TOTAL SCORE'].rank(pct=True),1)
        df['RATING'] = df['RANKING'].apply(lambda x:
            '⭐⭐⭐' if x >= .8 else (
            '⭐⭐' if x >= .5 else (
            '⭐' if x >=.2 else '')))
        df = df.sort_values(by='RANKING', ascending=False)

        return df

    def render(self, path, metadata, mode="browser"):
        
        self.path = path
        self.search_criteria=metadata
        self.mode = mode
        print(rank(self.path, self.search_criteria))
        # run_dash(self.path, self.search_criteria, self.mode)

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
        img_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(img_path)
        f.save(secure_filename(f.filename))
        analyzer = ResumeAnalyzer()
        search_criteria = {
            #experiences, education, projects, skills
            "Experiences":["neural networks", "cnn", "rnn", "ann", "lstm", "bert", "transformers", "algorithms"],
            "Education":["regression", "classification", "clustering", "time series", "summarization", "nlp"],
            "Projects/Research": ["Emotional","sarimax", "prophet", "holt winters"],
            "Skills": ["arima","sarimax", "techniques", "holt winters"],
            
        }
        df = rank(img_path, search_criteria)

        # leaf = leaf_predict(img_path)
        # if leaf == "leaf":
        #     # Make prediction
        #     preds = model_predict(img_path)
        #     rows = info()
        #     res = np.asarray(rows[preds])
        #     value = (preds == int(res[0]))
        #     if value:
        #         ID, Disease, Pathogen, Symptoms, Management = [i for i in res]
        #     return render_template('result.html', Pathogen=Pathogen, Symptoms=Symptoms, Management=Management, result=Disease, filee=f.filename)
        # else:
        #     return render_template('index.html', Error="ERROR: UPLOADED IMAGE IS NOT A LEAF (OR) MORE LEAVES IN ONE IMAGE")

        
        return render_template('result.html', Pathogen=df)
        # return result

@app.route('/predict/<filename>')
def send_file(filename):
    return send_from_directory('uploads', filename)



if __name__ == '__main__':
    app.run()


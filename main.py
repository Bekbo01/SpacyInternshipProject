from flask import Flask, render_template, request, jsonify
import pickle
import os
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import spacy

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
# nlp = spacy.load("en_core_web_trf")
nlp = spacy.load("output_updated/model-best")

model = AutoModelForTokenClassification.from_pretrained('old')
tokenizer = AutoTokenizer.from_pretrained('old')
ner_model = pipeline('ner', model=model, tokenizer=tokenizer)

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text


color_dict = {'JOB NAME': ["#17becf", "#9edae5"], 
              'JOB TIME': ["#9467bd", "#c5b0d5"], 
              'JOBTASKS': ["#e377c2", "#f7b6d2"], 
              'JOBREQUIREMENT': ["#74c476", "#c7e9c0"], 
              'JOB REQUIREMENT':  ["#74c476", "#c7e9c0"],
              'JOB TASKS': ["#e377c2", "#f7b6d2"],
             'none': ['dark']}

def load_models():
    models = dict()
    for file in os.listdir('./models/'):
        models[file.split('.')[0]] = pickle.load(open(f'models/{file}', 'rb'))
    return models

def predict_classic(my_txt, model, spacy_type=False):

    general = []
    for idx, j in enumerate([clean_text(i) for i in my_txt]):
        if spacy_type:
            try:
                pred = sorted(nlp(j).cats.items(), key= lambda x: x[1])[-1][0]
            except:
                pred = 'none'
        else:
            pred = model.predict([j])[0]
        if pred != 'none':
            general.append(f"""<span class="pred" style="background-color: {color_dict[pred][0]}"><span class="tooltip">{my_txt[idx]} {pred.upper()}</span></span>""")
        else:
            general.append(my_txt[idx])

    # my_tag_html = []
    # for k, v in color_dict.items():
    #     my_tag_html.append(f"""<span style="color: {v[0]}; font-weight: bold">{k}</span>""")

    # final_html = '\n'.join(my_tag_html) + '\n'
    final_html = '\n'.join(general)
    return final_html


def predict_ner(res, sequence):

    general = []
    if res[0]['start'] != 0:
        general.append(sequence[:res[0]['start']])
    for idx in range(len(res)):
        curr = res[idx]
        
        tag = curr['entity'].replace('-', ' ')
        pred_score = curr['score']
        dark, light = color_dict[tag]
        color = dark if pred_score >= 0.75 else light
        s = sequence[curr['start']:curr['end']]
        span = f"""<span class="pred" style="background-color: {color}"><span class="tooltip">{s}</span></span>"""
        general.append(span)
        
        if idx+1 < len(res):
            next_ = res[idx+1]
            if next_['start'] - curr['end'] > 0:
                general.append(sequence[curr['end']:next_['start']])
        # else:
        #     span = f"""<span style="color: {color}; font-weight: bold">{s}</span>"""

    my_tag_html = []
    for k, v in color_dict.items():
        my_tag_html.append(f"""<span style="color: {v[0]}; font-weight: bold">{k}</span>""")

    final_html = '\n'.join(my_tag_html) + '\n'
    final_html += ''.join(general)
    return final_html


models = load_models()
models['BERT'] = nlp
models['SPACY'] = ner_model
MODEL_NAMES = models.keys()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        textarea_value = request.form['textarea']
        select_value = request.form['selectbox']
        # print(type(select_value))
        if select_value == 'BERT':
            final_html = predict_ner(ner_model(textarea_value), textarea_value).replace('\n', '<br>')
        elif select_value == 'SPACY':
            my_txt = textarea_value.split('\n')
            final_html = predict_classic(my_txt, models[select_value], spacy_type=True).replace('\n', '<br>')
        else:
            my_txt = textarea_value.split('\n')
            final_html = predict_classic(my_txt, models[select_value]).replace('\n', '<br>')

        return jsonify({'result': final_html})
    return render_template('index.html', MODEL_NAMES=MODEL_NAMES)

if __name__ == '__main__':
    app.run(debug=True)

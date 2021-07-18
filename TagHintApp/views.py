# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:02:35 2021

@author: Fabretto
"""

import numpy as np
import pandas as pd

import os

from joblib import load

import regex
import nltk

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import TfidfVectorizer

# Import of my functions and module
import TagHintApp.Projets6Lib as prj6

from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from wtforms import TextAreaField

app = Flask(__name__.split('.')[0], instance_relative_config=True)
app.config.from_object('config')
# app.config.from_pyfile('config.py')
app.config['SECRET_KEY'] = 'Thisisasecret!'
#if os.environ.get('FLASK_ENV') == 'development':
#    app.config['SECRET_KEY'] = 'Thisisasecret!'

saved_model_path = os.path.join(app.static_folder, app.config['MODEL_FILE'])
    
with open(saved_model_path,'rb') as f:
    title_pipeline = load(f)
    tokenizer = load(f)
    t_vocabulary = load(f)
    tag_dictionary = load(f)

class MyForm(FlaskForm):

    title_text = TextAreaField('Title', 
                         validators=[DataRequired('A non empty title is required')])

@app.route('/', methods=['GET', 'POST'])
@app.route('/form', methods=['GET', 'POST'])
def form():
    form = MyForm()
    
    if form.validate_on_submit():
        input_tokens = prj6.tokenize_raw(form.title_text.data, tokenizer, t_vocabulary)
        return render_template('resultat.html', text=input_tokens)
    return render_template('form.html', form=form)

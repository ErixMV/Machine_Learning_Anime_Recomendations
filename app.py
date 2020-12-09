# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:41:37 2020

@author: Erix
"""

import os
from flask import Flask, jsonify, request as req
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



app = Flask(__name__)
app.debug = False

# a simple page that says hello
@app.route('/anime/recomendar/')
def hello():
    #result = test_procedure()
    username = req.args.get('title')
    valoration = int(req.args.get('valoration'))
    
    result = test_procedure(username, valoration)
    
    return result

@app.route('/anime/train')
def train():
    res = train_procedure()
    
    return jsonify({"result": res})

def getData():
    df_anime = pd.read_csv("./datos/anime.csv", sep=",", usecols=range(2), encoding='utf-8-sig')
    df_rating = pd.read_csv("./datos/rating.csv", encoding='utf-8-sig')[:100000]
    
    return df_rating.merge(df_anime)

def train_procedure():
    try:   
        df = getData()
        train, test = train_test_split(df, test_size= 0.2)
        
        userRatings = train.pivot_table(index=['user_id'],columns=['name'],values='rating')
        corrMatrix = userRatings.corr(method='pearson', min_periods=100)
        
        corrMatrix.to_pickle("./corrMatrix.pkl")

        return True
    except:
        return False
        
def test_procedure(username, valoration):
    try:
        ratingsSample = pd.Series({username: valoration})
        #ratingsSample = pd.Series({"Clannad": 9, "Elfen Lied": 10})
        
        corrMatrix = pd.read_pickle("./corrMatrix.pkl")
        
        simCandidates = pd.Series(dtype='float64')
        
        for i in range(0, len(ratingsSample.index)):
            # Recuperar las pelis similares a las calificadas
            sims = corrMatrix[ratingsSample.index[i]].dropna()
            # Escalar la similaridad multiplicando por la calificación de la persona
            sims = sims.map(lambda x: x * ratingsSample[i])
            # Añadir el puntaje a la lista de candidatos similares
            simCandidates = simCandidates.append(sims)
        
        #Mirar los resultados:
        simCandidates.sort_values(inplace = True, ascending = False)
        simCandidates = simCandidates.groupby(simCandidates.index).sum()
        simCandidates.sort_values(inplace = True, ascending = False)
        valoratedSeries = ratingsSample[(ratingsSample != -1.0)]
        filteredSims = simCandidates.drop(valoratedSeries.index)
        
        return filteredSims.to_json()  
    except Exception as inst:
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
                           # but may be overridden in exception subclasses
        #x, y = inst.args     # unpack ars
        #print('x =', x)
        #print('y =', y)
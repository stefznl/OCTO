import sys
import os
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# files
pathTrainFile = 'all/train_reduce.csv'
pathTestFile = 'all/test_reduce.csv'
pathSubmissionFileInput = 'all/sample_submission_reduce.csv'
pathSubmissionFileOutput = 'sample_submission_test.csv'

# variables
table = str.maketrans({key: None for key in string.punctuation})
alphabet = string.ascii_lowercase

def prepareString (s):
    """ Supprimer ponctuation et split de s """
    sNp = str(s).translate(table)
    sNpLower = sNp.lower()
    return sNp.split()

def prepMaj (s):
    """ Compte et indique les majuscules de s """
    cpt = 0
    listMaj = []
    for letter in str(s):
        if letter.isupper():
            cpt+=1
            listMaj.append(letter)
    return (cpt, listMaj)

def countLetter (myLetter, listWord):
    """ Compte une lettre (myLetter) dans une liste de mot(s) (listWord) """
    cpt = 0
    for word in listWord:
        if len(word) >= 4:
            for letter in word:
                if letter.lower() == myLetter:
                    cpt+=1
    return cpt

def creationDataFrameQuestion (data):
    """ Creation DataFrame from data Quora file """
    rdf = pd.DataFrame()
    question = ['question1', 'question2']
    for k, elem in enumerate(question):
        qdf = data[elem]
        rdf['words'+str(k)] = qdf.apply(lambda x: prepareString(x))
        rdf['NMaj'+str(k)] = qdf.apply(lambda x: prepMaj(x)[0])
        rdf['Maj'+str(k)] = qdf.apply(lambda x: prepMaj(x)[1])
        rdf['nmbreWord'+str(k)] = rdf['words'+str(k)].apply(lambda x: len(x))

        for letter in alphabet:
            rdf[str(letter)+str(k)] = rdf['words'+str(k)].apply(lambda x: countLetter(letter, x))    
    return rdf

def moyenneSimilarite (a1, b1):
    """ Moyenne ponderee de similarite entre deux nombres a1 et b1 """
    c = (max(a1, b1, 1))//(1+abs(a1-b1))
    return c

def compterElemSimiList (a, b):
    """ Moyenne ponderee de similarite entre deux listes a et b """
    cpt = 0
    for elem1 in a:
        j = 0
        notFound = True
        while (j < len(b))&(notFound):
            if elem1 == b[j]:
                cpt+=1
                notFound = False
            j+=1
    return cpt

def calibrateModel (df, step):
    """ Retourne la matrice X du DataFrame Quora df pour la step """
    print(step)
    rdf = creationDataFrameQuestion(df)

    X = pd.DataFrame()
    X['simiWord'] = rdf.apply( lambda data: compterElemSimiList(data['words0'],data['words1']) , axis=1)
    X['simiMajNumber'] = rdf.apply( lambda data: moyenneSimilarite(data['NMaj0'],data['NMaj1']) , axis=1 )
    X['simiMaj'] = rdf.apply( lambda data: compterElemSimiList(data['Maj0'],data['Maj1']) , axis=1 )

    for letter in alphabet:
        X['simi'+letter] = rdf.apply( lambda data: moyenneSimilarite(data[letter+'0'],data[letter+'1']) , axis=1 )
    return X

# import Data
df = pd.read_csv(pathTrainFile, dtype={'id' : int, 'qid1' : int, 'qid2' : int,'question1' : str,'question2' : str, 'is_duplicate' : int})
my_X = calibrateModel(df, 'begin train')
my_y = df.is_duplicate

# define models
from sklearn.ensemble import RandomForestClassifier
print('begin modeling')
model = RandomForestClassifier(random_state=1)
model.fit(my_X, my_y)

# prediction
print('begin prediction')
dftest = pd.read_csv(pathTestFile, dtype={'test_id' : int, 'question1' : str,'question2' : str})

# prerposs prediction
my_X_test = calibrateModel(dftest, 'begin test')
val_predict = model.predict(my_X_test)

# output pred
dftest = pd.read_csv(pathSubmissionFileInput)
dftest['is_duplicate'] = val_predict
dftest['is_duplicate'] = dftest['is_duplicate'].apply(lambda x: int(x))

# make kaggle file
dftest.to_csv(pathSubmissionFileOutput, index = False)
#!/usr/bin/env python
# coding: utf-8

# In[24]:


from flask import Flask, request, jsonify, request
import joblib
from joblib import load


app = Flask(__name__)


modelo_arbol_A = joblib.load('modelo_arbol_A.joblib')
modelo_arbol_B = joblib.load('modelo_arbol_B.joblib')

etiquetas_clasificacion = ['No diabetico', 'Prediabetico', 'Diabetico']

@app.route('/prediccionAnalitica', methods=['GET'])
def predict_analitica():
    
    a1 = float(request.args.get('a1'))
    a2 = float(request.args.get('a2'))
    a3 = float(request.args.get('a3'))
    a4 = float(request.args.get('a4'))
    a5 = float(request.args.get('a5'))
    a6 = float(request.args.get('a6'))
    a7 = float(request.args.get('a7'))
    a8 = float(request.args.get('a8'))
    a9 = float(request.args.get('a9'))
    a10 = float(request.args.get('a10'))
    a11 = float(request.args.get('a11'))

    datos_analitica = list([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11])

    print(datos_analitica)
    
    prediccion = modelo_arbol_A.predict_proba([datos_analitica])[0]
    for i in range(len(prediccion)):
        if prediccion[i]==max(prediccion):
            indice=i
    label = etiquetas_clasificacion[indice]
    resul = [label, max(prediccion)*100]
    
    return jsonify(status='Prediccion completada', prediccion = resul)


@app.route('/prediccionHabitosVida', methods=['GET'])
def predict_habitos_vida():
    
    h1 = float(request.args.get('h1'))
    h2 = float(request.args.get('h2'))
    h3 = float(request.args.get('h3'))
    h4 = float(request.args.get('h4'))
    h5 = float(request.args.get('h5'))
    h6 = float(request.args.get('h6'))
    h7 = float(request.args.get('h7'))
    h8 = float(request.args.get('h8'))
    h9 = float(request.args.get('h9'))
    h10 = float(request.args.get('h10'))
    h11 = float(request.args.get('h11'))
    h12 = float(request.args.get('h12'))
    h13 = float(request.args.get('h13'))
    h14 = float(request.args.get('h14'))
    h15 = float(request.args.get('h15'))
    h16 = float(request.args.get('h16'))
    h17 = float(request.args.get('h17'))
    h18 = float(request.args.get('h18'))
    h19 = float(request.args.get('h19'))
    h20 = float(request.args.get('h20'))
    h21 = float(request.args.get('h21'))


    datos_analitica = list([h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21])

    print(datos_analitica)
    
    prediccion = modelo_arbol_B.predict_proba([datos_analitica])[0]
    for i in range(len(prediccion)):
        if prediccion[i]==max(prediccion):
            indice=i
    label = etiquetas_clasificacion[indice]
    resul = [label, max(prediccion)*100]
    
    return jsonify(status='Prediccion completada', prediccion = resul)



    


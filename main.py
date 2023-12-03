import pandas as pd
import numpy as np
import pickle
import json
import config
from utilities import image_load_and_preprocess

from flask import Flask,render_template,request,jsonify

app = Flask(__name__)

with open(config.model_path, "rb") as file:
    model = pickle.load(file)

with open(config.tfidf_path, "rb") as file1:
    tfidf = pickle.load(file1)

# Class Labels
label_data = ['email', 'resume' ,'scientific_publication']

@app.route("/")
def home_page():
    return render_template("index.html") #"Welcome To Home App"

@app.route("/prediction", methods=["POST"])
def document_prediction():
    if request.method=="POST":
        #image_file = request.files['file']
        
        # image_path = 'temp_image.png'
        # image_file.save(image_path)

        # text_data = image_load_and_preprocess(image_file)
        text_data = request.form['text']

        # TFIDF
        tfidf_data = tfidf.transform([text_data])
        # Model
        y_pred = model.predict(tfidf_data.A)

        if y_pred==0:
            result = "This Document is E-mail"
        elif y_pred==1:
            result = "This Document is Resume"
        elif y_pred==2:
            result = "This Document is Scientific Publication"
        
        return render_template("result.html",  output=result)
    
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True, port=config.PORT, host=config.HOST)

#<button type="submit">Classify Document</button>










# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Document</title>
# </head>
# <body bgcolor="plum">
#     <h3>Document Classification Model</h3>
#     <form method="POST" action="/prediction">
#         <label for="text">Upload Document Image:</label>

#         <input type="file" name="file" accept=".png, .jpg, .jpeg" required>
#         <br>
#         <br>
#         <input type="submit" value="Classify Document">


#     </form>
    
# </body>
# </html>





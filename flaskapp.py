from flask import Flask, jsonify, request
from Classifier import alphabet_pred_boy

app = Flask(__name__)

@app.route('/prediction', methods = ['POST'])
def call_preded_alphabet():
    img = request.files.get('image')
    imgpred = alphabet_pred_boy(img)
    return jsonify({
        'prediction':imgpred
    })

if (__name__ == '__main__'):
    app.run(debug=True)
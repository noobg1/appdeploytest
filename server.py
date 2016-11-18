from sentimentEngine import sentiment
from flask import Flask, Response
import json
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


app = Flask(__name__)


@app.route('/')
def start():
	return "working"


@app.route('/q/<text>')
def query(text):
	result = (sentiment(tokenizer.tokenize(text)))
	data = {'result'  : str(result) }
	js = json.dumps(data)
	resp = Response(js, status=200, mimetype='application/json')
	return resp
	
if __name__ == '__main__':
    app.run(debug=True,port=5000)
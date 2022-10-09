from chat import ChatBot
from flask import Flask, render_template, request, session
# from flask_session import Session
import pandas as pd
import random
import secrets

app = Flask(__name__)

v_prompt = "VidyaBot is a chatbot that is the digital recreation of a dead woman designed to help her friends grieve, she is upbeat but spooky.\n"
# comps = pd.read_pickle('/home/pbrehill/VidyaBot/VidyaBot2/embeddings.pkl')
comps = pd.read_pickle('embeddings.pkl')

secret = secrets.token_urlsafe(32)

app.secret_key = secret

chatbot = ChatBot(comps)

# SESSION_TYPE = 'redis'
# Session(app)

@app.route("/")
def home():
    session['key'] = "".join([random.choice('0123456789ABCDEF') for i in range(16)])
    session['orig_prompt'] = v_prompt
    session['prompt'] = v_prompt
    session['i'] = 0
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    resp, prompt = chatbot.get_reply(session['prompt'] + '\nHuman: ' + (userText), session['orig_prompt'], session['i'])
    session['prompt'] = prompt
    session['i'] += 1
    return resp.replace("VidyaBot: ", "")

if __name__ == "__main__":
    app.run()
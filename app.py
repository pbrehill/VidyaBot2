from chat import ChatBot
from flask import Flask, render_template, request, session
from flask.ext.session import Session
import pandas as pd
app = Flask(__name__)

v_prompt = "VidyaBot is a chatbot that is the digital recreation of a dead woman designed to help her friends grieve, she is upbeat but spooky.\n"
comps = pd.read_pickle('/home/pbrehill/VidyaBot/VidyaBot2/embeddings.pkl')

chatbot = ChatBot(v_prompt, comps)

SESSION_TYPE = 'redis'
Session(app)

@app.route("/")
def home():
    session['key'] = "".join([random.choice('0123456789ABCDEF') for i in range(16)])
    session['v_prompt'] = v_prompt
    session['prompt'] = v_prompt
    return render_template("index.html")

@app.route("/get")
def get_bot_response(chatbot):
    userText = request.args.get('msg')
    resp = str(chatbot.get_replysession['prompt'] + (userText))
    session['prompt'] = resp
    return resp.replace("VidyaBot: ", "")

if __name__ == "__main__":
    app.run()
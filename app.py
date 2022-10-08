from chat import ChatBot
from flask import Flask, render_template, request
import pandas as pd
app = Flask(__name__)

v_prompt = "VidyaBot is a chatbot that is the digital recreation of a dead woman designed to help her friends grieve, she is upbeat but spooky.\n"
comps = pd.read_pickle('embeddings.pkl')

chatbot = ChatBot(v_prompt, comps)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    resp = str(chatbot.get_reply(userText))
    return (resp)

if __name__ == "__main__":
    app.run()
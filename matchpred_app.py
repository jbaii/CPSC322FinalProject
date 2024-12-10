import pickle
# we are going to use the Flask micro web framework
from flask import Flask, request, jsonify
from joblib import dump, load
import numpy as np
from mysklearn.myclassifiers import *
app = Flask(__name__)

def load_model():
    # unpickle header and tree in tree.p
    model = load("RF_math.joblib")
    return model
# we need to add some routes!
# a "route" is a function that handles a request
# e.g. for the HTML content for a home page
# or for the JSON response for a /predict API endpoint, etc
@app.route("/")
def index():
    # return content and status code
    return "<h1>Welcome to the interview predictor app</h1>", 200
        

# lets add a route for the /predict endpoint
@app.route("/predict")
def predict():
    # lets parse the unseen instance values from the query string
    # they are in the request object
    home_rank = request.args.get("home_team_rank") # defaults to None
    away_rank = request.args.get("away_team_rank")
    instance = [home_rank, away_rank]
    model = load_model()
    # lets make a prediction!
    pred = model.predict([np.array([float(home_rank), float(away_rank)])])
    if pred is not None:
        return jsonify({"prediction": pred}), 200
    # something went wrong!!
    # print(pred)




if __name__ == "__main__":
    # header, tree = load_model()
    # print(header)
    # print(tree)
    app.run(host="0.0.0.0", port=5001, debug=False)
    # TODO: when deploy app to "production", set debug=False
    # and check host and port values

    # instructions for deploying flask app to render.com: https://docs.render.com/deploy-flask
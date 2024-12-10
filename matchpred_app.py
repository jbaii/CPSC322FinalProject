from flask import Flask, request, render_template, jsonify
from joblib import load
import numpy as np

app = Flask(__name__)

def load_model():
    # 加载预训练模型
    return load("RF_math.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # 获取两个输入
            home_team_rank = float(request.form.get("home_team_rank", 0))
            away_team_rank = float(request.form.get("away_team_rank", 0))
            
            # 预测
            model = load_model()
            pred = model.predict([[home_team_rank, away_team_rank]])
            prediction = pred[0]  # 假设模型返回的是一个列表
        except Exception as e:
            prediction = f"Error: {str(e)}"
        print(prediction)
    return render_template("./index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

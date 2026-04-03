from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("dashboard.html")

@app.route('/api/data')
def get_data():
    if os.path.exists('dashboard.json'):
        with open('dashboard.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({"error": "Data not found. Run export_dashboard_data.py first."}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)

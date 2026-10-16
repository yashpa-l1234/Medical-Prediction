import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained Linear Regression model
with open('linear1.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML + Tailwind CSS Template for a modern, attractive UI
HOME_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Cost Prediction Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-purple-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-800/80 backdrop-blur-md rounded-2xl shadow-2xl border border-slate-700 p-8">
        <div class="text-center mb-8">
            <span class="bg-indigo-500/10 text-indigo-400 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider border border-indigo-500/20">AI Prediction Model</span>
            <h1 class="text-3xl font-extrabold mt-3 tracking-tight text-white">Cost & Charge Predictor</h1>
            <p class="text-slate-400 text-sm mt-1">Enter your details below to calculate the estimated medical charges.</p>
        </div>

        <form method="POST" action="/" class="space-y-6">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Age -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Age</label>
                    <input type="number" name="age" required min="1" max="120" placeholder="e.g., 30" 
                        class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">BMI (Body Mass Index)</label>
                    <input type="number" step="0.1" name="bmi" required min="10" max="60" placeholder="e.g., 25.5" 
                        class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- Children -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Number of Children</label>
                    <input type="number" name="children" required min="0" max="10" placeholder="e.g., 2" 
                        class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- Sex -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Sex</label>
                    <select name="sex" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="1">Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>

                <!-- Smoker -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Smoker Status</label>
                    <select name="smoker" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                    </select>
                </div>

                <!-- Region -->
                <div>
                    <label class="block text-sm font-medium text-slate-300 mb-2">Region</label>
                    <select name="region" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="0">Northeast</option>
                        <option value="1">Northwest</option>
                        <option value="2">Southeast</option>
                        <option value="3">Southwest</option>
                    </select>
                </div>
            </div>

            <button type="submit" class="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3.5 rounded-lg shadow-lg hover:shadow-indigo-500/25 transition duration-200">
                Calculate Prediction
            </button>
        </form>

        {% if prediction_text %}
        <div class="mt-8 p-6 bg-indigo-950/50 border border-indigo-500/30 rounded-xl text-center animate-fade-in">
            <h2 class="text-sm font-medium text-indigo-300 uppercase tracking-wider">Estimated Result</h2>
            <p class="text-4xl font-extrabold text-white mt-2">{{ prediction_text }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def home():
  prediction_text = None
  if request.method == 'POST':
    try:
      # Extract form inputs
      age = float(request.form['age'])
      sex = float(request.form['sex'])
      bmi = float(request.form['bmi'])
      children = float(request.form['children'])
      smoker = float(request.form['smoker'])
      region = float(request.form['region'])

      # Create feature array matching model.feature_names_in_ order
      # Features: ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
      features = np.array([[age, sex, bmi, children, smoker, region]])

      # Make prediction
      prediction = model.predict(features)[0]
      prediction_text = f"${prediction:,.2f}"
    except Exception as e:
      prediction_text = f'Error in calculation: {str(e)}'

  return render_template_string(
      HOME_PAGE_HTML, prediction_text=prediction_text
  )


if __name__ == '__main__':
  app.run(debug=True)

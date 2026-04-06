## Disease Prediction Web Application

Flask-based web application that predicts possible diseases based on user-selected symptoms.  
The app includes full user authentication (registration, login, profile management) and uses a pre-trained ML model together with several medical datasets to display descriptions, precautions, diet, medications, and workout recommendations for the predicted disease.

### Features
- **User authentication**
  - Registration with password strength validation and email uniqueness checks
  - Login / logout via `flask-login`
  - First-time **profile completion** (gender, age, phone) before using the app
  - Profile update page (change basic info and password)
- **Disease prediction**
  - Symptom checklist generated from `data/Diseases_and_Symptoms_dataset.csv`
  - Uses a trained model loaded from `model/disease_prediction_model.pkl`
  - Optional prediction confidence (if `predict_proba` is available)
- **Rich disease information**
  - Description from `data/description.csv`
  - Precautions from `data/precautions.csv`
  - Diet recommendations from `data/diets.csv`
  - Medication suggestions from `data/medications.csv`
  - Workout / lifestyle tips from `data/workout.csv`
- **Web UI**
  - HTML templates in `templates/`
  - Central styling in `static/style.css`

### Tech Stack
- **Backend**: Python, Flask
- **Auth & DB**: `flask-login`, `Flask-SQLAlchemy`, SQLite (`instance/users.db`)
- **ML / Data**: `joblib`, `pandas`, `numpy`

### Project Structure (simplified)
- `app.py` – main Flask application (routes, model loading, prediction logic, auth)
- `requirements.txt` – Python dependencies
- `model/disease_prediction_model.pkl` – trained ML model
- `data/` – CSV files for symptoms, descriptions, precautions, diet, medication, workout
- `templates/` – HTML templates (login, register, profile, index, result, etc.)
- `static/style.css` – CSS styles

### Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd "Final Project"
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model and data files are present**
   - `model/disease_prediction_model.pkl`
   - CSV files inside the `data/` folder:
     - `Diseases_and_Symptoms_dataset.csv`
     - `description.csv`
     - `precautions.csv`
     - `diets.csv`
     - `medications.csv`
     - `workout.csv`

5. **Run the application**
   ```bash
   python app.py
   ```
   The Flask app (by default) runs on `http://127.0.0.1:5000/` with `debug=True`.

### How to Use
- Open the app in your browser.
- **Register** a new account.
- **Log in**, then complete your profile (gender, age, phone) on first login.
- Go to the **home page**, select your symptoms from the checklist, and submit.
- View:
  - Predicted disease
  - Description
  - Precautions
  - Diet suggestions
  - Medication information
  - Workout / lifestyle recommendations

## ⚠️ Disclaimer

Educational purpose only.

---

## 👤 Author

Sankalp S Nair  
https://github.com/SankalpSNair


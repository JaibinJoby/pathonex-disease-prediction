"""
Disease Prediction Web Application with User Authentication
Flask app for predicting diseases based on symptoms with user management
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'  # Change this in production!
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Configuration
MODEL_PATH = 'model/disease_prediction_model.pkl'
DATASET_PATH = 'data/Diseases_and_Symptoms_dataset.csv'
DESCRIPTION_PATH = 'data/description.csv'
PRECAUTIONS_PATH = 'data/precautions.csv'
DIETS_PATH = 'data/diets.csv'
MEDICATIONS_PATH = 'data/medications.csv'
WORKOUT_PATH = 'data/workout.csv'

# Global variables to store loaded data
model = None
symptom_names = []
disease_data = {}


# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=True)  # Will be set during profile completion
    password_hash = db.Column(db.String(255), nullable=False)
    gender = db.Column(db.String(10), nullable=True)  # Male, Female, Other
    age = db.Column(db.Integer, nullable=True)
    profile_completed = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def get_disease_info(disease_name, data_dict):
    """
    Helper function to retrieve disease information with case-insensitive matching.
    Returns the value if found, otherwise returns None.
    """
    # Try exact match first
    if disease_name in data_dict:
        return data_dict[disease_name]
    
    # Try case-insensitive match
    disease_lower = disease_name.lower()
    for key, value in data_dict.items():
        if key.lower() == disease_lower:
            return value
    
    return None


def load_model_and_data():
    """
    Load the trained model and all CSV data files.
    This function is called once when the app starts.
    """
    global model, symptom_names, disease_data
    
    try:
        # Load the trained model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            # Get expected number of features from model
            expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
            if expected_features:
                print(f"Model expects {expected_features} features")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
            print("Please ensure disease_prediction_model.pkl is in the model/ directory")
            return False
        
        # Load the main dataset to extract symptom names
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            # Extract symptom column names (exclude disease column - first column)
            # Also check for common variations: 'Disease', 'diseases', 'Diseases'
            disease_col_variations = ['Disease', 'diseases', 'Diseases', 'disease']
            symptom_names = [col for col in df.columns if col not in disease_col_variations]
            # If still wrong count, exclude first column (most reliable method)
            if len(symptom_names) != len(df.columns) - 1:
                symptom_names = df.columns[1:].tolist()
            
            # Validate that symptom count matches model's expected features
            if expected_features and len(symptom_names) != expected_features:
                print(f"ERROR: Mismatch detected!")
                print(f"  Model expects: {expected_features} features")
                print(f"  Dataset has: {len(symptom_names)} symptom columns")
                print(f"  Total columns in dataset: {len(df.columns)}")
                print(f"  First column (excluded): {df.columns[0]}")
                # Force exclude first column to match model
                symptom_names = df.columns[1:].tolist()
                print(f"  Fixed: Now using {len(symptom_names)} symptoms (excluding first column)")
            
            print(f"Loaded {len(symptom_names)} symptoms from dataset")
            print(f"Excluded column(s): {set(df.columns) - set(symptom_names)}")
        else:
            print(f"Warning: Dataset file not found at {DATASET_PATH}")
            return False
        
        # Load disease-related data files
        disease_data = {}
        
        # Load description
        if os.path.exists(DESCRIPTION_PATH):
            desc_df = pd.read_csv(DESCRIPTION_PATH)
            disease_data['description'] = dict(zip(desc_df.iloc[:, 0], desc_df.iloc[:, 1]))
        else:
            print(f"Warning: Description file not found at {DESCRIPTION_PATH}")
            disease_data['description'] = {}
        
        # Load precautions
        if os.path.exists(PRECAUTIONS_PATH):
            prec_df = pd.read_csv(PRECAUTIONS_PATH)
            # Assuming first column is disease name, rest are precautions
            disease_data['precautions'] = {}
            for idx, row in prec_df.iterrows():
                disease = row.iloc[0]
                precautions = [str(val) for val in row.iloc[1:].dropna().tolist() if str(val).strip()]
                disease_data['precautions'][disease] = precautions
        else:
            print(f"Warning: Precautions file not found at {PRECAUTIONS_PATH}")
            disease_data['precautions'] = {}
        
        # Load diets
        if os.path.exists(DIETS_PATH):
            diet_df = pd.read_csv(DIETS_PATH)
            disease_data['diet'] = dict(zip(diet_df.iloc[:, 0], diet_df.iloc[:, 1]))
        else:
            print(f"Warning: Diets file not found at {DIETS_PATH}")
            disease_data['diet'] = {}
        
        # Load medications
        if os.path.exists(MEDICATIONS_PATH):
            med_df = pd.read_csv(MEDICATIONS_PATH)
            disease_data['medication'] = dict(zip(med_df.iloc[:, 0], med_df.iloc[:, 1]))
        else:
            print(f"Warning: Medications file not found at {MEDICATIONS_PATH}")
            disease_data['medication'] = {}
        
        # Load workout
        if os.path.exists(WORKOUT_PATH):
            workout_df = pd.read_csv(WORKOUT_PATH)
            disease_data['workout'] = dict(zip(workout_df.iloc[:, 0], workout_df.iloc[:, 1]))
        else:
            print(f"Warning: Workout file not found at {WORKOUT_PATH}")
            disease_data['workout'] = {}
        
        print("All data loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model or data: {str(e)}")
        return False


# AJAX Validation Routes
@app.route('/api/check-email', methods=['POST'])
def check_email():
    """AJAX endpoint to check if email is available"""
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    
    if not email:
        return {'available': False, 'message': 'Email is required'}, 400
    
    # Validate email format
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return {'available': False, 'message': 'Invalid email format'}, 400
    
    # Check if email exists
    user = User.query.filter_by(email=email).first()
    if user:
        return {'available': False, 'message': 'Email already registered'}, 200
    else:
        return {'available': True, 'message': 'Email is available'}, 200


@app.route('/api/validate-password', methods=['POST'])
def validate_password():
    """AJAX endpoint to validate password strength"""
    data = request.get_json()
    password = data.get('password', '')
    
    if not password:
        return {'valid': False, 'message': 'Password is required', 'strength': 0}, 400
    
    strength = 0
    messages = []
    
    # Length check
    if len(password) >= 6:
        strength += 1
    else:
        messages.append('At least 6 characters')
    
    # Uppercase check
    if any(c.isupper() for c in password):
        strength += 1
    else:
        messages.append('One uppercase letter')
    
    # Lowercase check
    if any(c.islower() for c in password):
        strength += 1
    else:
        messages.append('One lowercase letter')
    
    # Number check
    if any(c.isdigit() for c in password):
        strength += 1
    else:
        messages.append('One number')
    
    # Special character check
    special_chars = '!@#$%^&*()_+-=[]{}|;:,.<>?'
    if any(c in special_chars for c in password):
        strength += 1
    else:
        messages.append('One special character')
    
    # Length >= 8
    if len(password) >= 8:
        strength += 1
    
    valid = len(password) >= 6
    strength_level = ['weak', 'fair', 'good', 'strong', 'very strong'][min(strength - 1, 4)] if strength > 0 else 'weak'
    
    if valid:
        return {
            'valid': True,
            'message': f'Password strength: {strength_level}',
            'strength': strength,
            'strength_level': strength_level,
            'suggestions': messages if strength < 3 else []
        }, 200
    else:
        return {
            'valid': False,
            'message': 'Password must be at least 6 characters',
            'strength': 0,
            'strength_level': 'weak',
            'suggestions': messages
        }, 200


@app.route('/api/validate-phone', methods=['POST'])
def validate_phone():
    """AJAX endpoint to validate phone number"""
    data = request.get_json()
    phone = data.get('phone', '').strip()
    
    if not phone:
        return {'valid': False, 'message': 'Phone number is required'}, 400
    
    # Basic phone validation (digits, spaces, dashes, parentheses, plus)
    import re
    phone_pattern = r'^[\d\s\-\+\(\)]{10,}$'
    cleaned_phone = re.sub(r'[\s\-\+\(\)]', '', phone)
    
    if not re.match(phone_pattern, phone):
        return {'valid': False, 'message': 'Invalid phone number format'}, 200
    
    if len(cleaned_phone) < 10:
        return {'valid': False, 'message': 'Phone number must be at least 10 digits'}, 200
    
    if len(cleaned_phone) > 15:
        return {'valid': False, 'message': 'Phone number is too long'}, 200
    
    return {'valid': True, 'message': 'Valid phone number'}, 200


# Authentication Routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Validation
        errors = []
        if not full_name:
            errors.append('Full name is required.')
        if not email:
            errors.append('Email is required.')
        elif User.query.filter_by(email=email).first():
            errors.append('Email already registered.')
        if not password:
            errors.append('Password is required.')
        elif len(password) < 6:
            errors.append('Password must be at least 6 characters long.')
        if password != confirm_password:
            errors.append('Passwords do not match.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
        
        # Create new user
        try:
            new_user = User(
                full_name=full_name,
                email=email
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred during registration. Please try again.', 'error')
            print(f"Registration error: {str(e)}")
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please enter both email and password.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.full_name}!', 'success')
            
            # Check if profile needs completion
            if not user.profile_completed:
                return redirect(url_for('complete_profile'))
            
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))


@app.route('/complete-profile', methods=['GET', 'POST'])
@login_required
def complete_profile():
    """Profile completion route - gender and age (first login only)"""
    if current_user.profile_completed:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        gender = request.form.get('gender', '').strip()
        age = request.form.get('age', '').strip()
        phone = request.form.get('phone', '').strip()
        
        errors = []
        if not gender:
            errors.append('Gender is required.')
        elif gender not in ['Male', 'Female', 'Other']:
            errors.append('Invalid gender selection.')
        if not age:
            errors.append('Age is required.')
        else:
            try:
                age_int = int(age)
                if age_int < 1 or age_int > 120:
                    errors.append('Please enter a valid age.')
            except ValueError:
                errors.append('Age must be a number.')
        if not phone:
            errors.append('Phone number is required.')
        else:
            # Basic phone validation
            import re
            phone_pattern = r'^[\d\s\-\+\(\)]{10,}$'
            cleaned_phone = re.sub(r'[\s\-\+\(\)]', '', phone)
            if not re.match(phone_pattern, phone):
                errors.append('Invalid phone number format.')
            elif len(cleaned_phone) < 10:
                errors.append('Phone number must be at least 10 digits.')
            elif len(cleaned_phone) > 15:
                errors.append('Phone number is too long.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('complete_profile.html')
        
        # Update user profile
        try:
            current_user.gender = gender
            current_user.age = int(age)
            current_user.phone = phone
            current_user.profile_completed = True
            db.session.commit()
            
            flash('Profile completed successfully!', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            print(f"Profile completion error: {str(e)}")
    
    return render_template('complete_profile.html')


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    """User profile view and update route"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip().lower()
        phone = request.form.get('phone', '').strip()
        gender = request.form.get('gender', '').strip()
        age = request.form.get('age', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        errors = []
        
        # Validate required fields
        if not full_name:
            errors.append('Full name is required.')
        if not email:
            errors.append('Email is required.')
        if not phone:
            errors.append('Phone number is required.')
        
        # Check if email is already taken by another user
        existing_user = User.query.filter_by(email=email).first()
        if existing_user and existing_user.id != current_user.id:
            errors.append('Email already registered to another account.')
        
        # Validate gender if provided
        if gender and gender not in ['Male', 'Female', 'Other']:
            errors.append('Invalid gender selection.')
        
        # Validate age if provided
        if age:
            try:
                age_int = int(age)
                if age_int < 1 or age_int > 120:
                    errors.append('Please enter a valid age.')
            except ValueError:
                errors.append('Age must be a number.')
        
        # Validate password if provided
        if password:
            if len(password) < 6:
                errors.append('Password must be at least 6 characters long.')
            if password != confirm_password:
                errors.append('Passwords do not match.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('profile.html', user=current_user)
        
        # Update user information
        try:
            current_user.full_name = full_name
            current_user.email = email
            current_user.phone = phone
            if gender:
                current_user.gender = gender
            if age:
                current_user.age = int(age)
            if password:
                current_user.set_password(password)
            current_user.updated_at = datetime.utcnow()
            db.session.commit()
            
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'error')
            print(f"Profile update error: {str(e)}")
    
    return render_template('profile.html', user=current_user)


@app.route('/about')
@login_required
def about():
    """About page route - explains ML-based predictions and medical disclaimer"""
    return render_template('about.html', user=current_user)


# Main Application Routes
@app.route('/')
@login_required
def index():
    """
    Home page route - displays symptom selection form
    Requires login and profile completion
    """
    if not current_user.profile_completed:
        return redirect(url_for('complete_profile'))
    
    return render_template('index.html', symptoms=symptom_names, user=current_user)


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Prediction route - handles symptom submission and returns prediction
    Requires login and profile completion
    """
    if not current_user.profile_completed:
        return redirect(url_for('complete_profile'))
    
    try:
        # Get selected symptoms from form
        selected_symptoms = request.form.getlist('symptoms')
        
        if not selected_symptoms:
            # No symptoms selected
            return render_template('result.html', 
                                 error="Please select at least one symptom.",
                                 symptoms=symptom_names,
                                 user=current_user)
        
        # Create binary input vector matching model training format
        # Initialize with zeros for all symptoms
        input_vector = np.zeros(len(symptom_names))
        
        # Set selected symptoms to 1
        for symptom in selected_symptoms:
            if symptom in symptom_names:
                symptom_index = symptom_names.index(symptom)
                input_vector[symptom_index] = 1
        
        # Reshape for model prediction (model expects 2D array)
        input_vector = input_vector.reshape(1, -1)
        
        # Predict disease
        predicted_disease = model.predict(input_vector)[0]
        
        # Get probability scores (if available)
        try:
            probabilities = model.predict_proba(input_vector)[0]
            disease_classes = model.classes_
            confidence = max(probabilities) * 100
        except:
            confidence = None
            disease_classes = None
        
        # Fetch disease-related information with case-insensitive matching
        description = get_disease_info(predicted_disease, disease_data.get('description', {}))
        if description is None:
            description = 'No description available.'
        
        precautions = get_disease_info(predicted_disease, disease_data.get('precautions', {}))
        if precautions is None:
            precautions = []
        
        diet = get_disease_info(predicted_disease, disease_data.get('diet', {}))
        if diet is None:
            diet = 'No specific diet recommendation available.'
        
        medication = get_disease_info(predicted_disease, disease_data.get('medication', {}))
        if medication is None:
            medication = 'No medication information available.'
        
        workout = get_disease_info(predicted_disease, disease_data.get('workout', {}))
        if workout is None:
            workout = 'No workout/lifestyle recommendation available.'
        
        # Render result page
        return render_template('result.html',
                             disease=predicted_disease,
                             description=description,
                             precautions=precautions,
                             diet=diet,
                             medication=medication,
                             workout=workout,
                             confidence=confidence,
                             selected_symptoms=selected_symptoms,
                             user=current_user)
    
    except Exception as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        print(error_message)
        return render_template('result.html', 
                             error=error_message,
                             symptoms=symptom_names,
                             user=current_user)


if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        try:
            # Try to create tables (will not fail if they exist)
            db.create_all()
            print("Database initialized.")
        except Exception as e:
            print(f"Database initialization note: {str(e)}")
            print("If you see constraint errors, run: python migrate_database.py")
    
    # Load model and data on startup
    print("Initializing Disease Prediction Web Application...")
    if load_model_and_data():
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize application. Please check your model and data files.")

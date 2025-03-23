#import generativeai
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_pymongo import PyMongo
from pymongo import MongoClient
from datetime import datetime, timedelta, timezone
import pytz
import json
from bson import json_util
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
import random
from bson import ObjectId
import google.generativeai as genai
from PIL import Image
import io
import base64
import os
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = 'your-secret-key-123'

try:
    app.config["MONGO_URI"] = os.getenv("MONGODB_URI")
    if not app.config["MONGO_URI"]:
        raise ValueError("MONGODB_URI environment variable is not set")

    # Initialize MongoDB connection with retry writes and TLS settings
    mongo = PyMongo(app)

    # Test connection
    mongo.db.command('ping')
    print("✅ MongoDB connection successful!")
except Exception as e:
    print(f"❌ MongoDB connection error: {str(e)}")
    raise

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.user_data = user_data

    def get_id(self):
        return str(self.user_data['_id'])

    @property
    def name(self):
        return self.user_data.get('name', 'User')


# Helper function for MongoDB operations
def execute_mongo_operation(operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        print(f"MongoDB operation error: {str(e)}")
        return None


# Example of using the helper function in the user loader
@login_manager.user_loader
def load_user(user_id):
    user_data = execute_mongo_operation(mongo.db.users.find_one, {'_id': ObjectId(user_id)})
    return User(user_data) if user_data else None


# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = mongo.db.users.find_one({'email': email})

        if user and check_password_hash(user['password'], password):
            login_user(User(user))
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        if mongo.db.users.find_one({'email': email}):
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({
            'name': name,
            'email': email,
            'password': hashed_password,
            'created_at': datetime.now(pytz.timezone('Asia/Kolkata'))
        })

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    try:
        data = request.form
        user_id = current_user.get_id()

        # Validate the data
        if not data.get('name') or not data.get('email'):
            flash('Name and email are required', 'error')
            return redirect(url_for('profile'))

        # Update user data
        update_data = {
            'name': data.get('name'),
            'email': data.get('email')
        }

        # Handle password update if provided
        new_password = data.get('new_password')
        if new_password:
            if new_password != data.get('confirm_password'):
                flash('Passwords do not match', 'error')
                return redirect(url_for('profile'))
            update_data['password'] = generate_password_hash(new_password)

        # Update the user in database
        result = mongo.db.users.update_one(
            {'_id': user_id},
            {'$set': update_data}
        )

        if result.modified_count > 0:
            flash('Profile updated successfully', 'success')
        else:
            flash('No changes were made', 'info')

        return redirect(url_for('profile'))

    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        flash('An error occurred while updating your profile', 'error')
        return redirect(url_for('profile'))


# Ensure required collections exist
try:
    db = mongo.db
    energy_data = db.energy_usage
    food_items = db.food_items
    deliveries = db.deliveries
    impact_stats = db.impact_stats
    readings = db.readings
    food_waste = db.food_waste
    waste_stats = db.waste_stats
    carbon_activities = db.carbon_activities
    users = db.users
except Exception as e:
    print(f"MongoDB connection error: {e}")

# Appliance power ranges in kWh per hour
APPLIANCE_POWER = {
    'AC': {'min': 1.0, 'max': 4.0, 'typical': 2.5},
    'Refrigerator': {'min': 0.1, 'max': 0.5, 'typical': 0.2},
    'Washing Machine': {'min': 0.4, 'max': 1.5, 'typical': 0.8},
    'TV': {'min': 0.1, 'max': 0.4, 'typical': 0.2},
    'Lights': {'min': 0.02, 'max': 0.2, 'typical': 0.06},
    'Computer': {'min': 0.1, 'max': 0.5, 'typical': 0.2},
    'Microwave': {'min': 0.6, 'max': 1.5, 'typical': 1.0},
    'Water Heater': {'min': 1.5, 'max': 4.0, 'typical': 2.5},
    'Fan': {'min': 0.05, 'max': 0.2, 'typical': 0.1},
    'Other': {'min': 0.05, 'max': 3.0, 'typical': 0.5}
}

# Set timezone to IST
IST = pytz.timezone('Asia/Kolkata')


def get_current_time():
    """Get current time in IST"""
    return datetime.now(IST)



# Configure Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# List and print available models
try:
    available_models = genai.list_models()
    print("\nAvailable Gemini models:")
    for model in available_models:
        print(f"- {model.name} ({model.supported_generation_methods})")
except Exception as e:
    print(f"Error listing Gemini models: {e}")

# Initialize Gemini model
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")


# Main routes
@app.route('/')
def index():
    try:
        if current_user.is_authenticated:
            user_id = current_user.get_id()

        # Get platform stats
        total_users = mongo.db.users.count_documents({})

        # Calculate total energy saved
        energy_saved = mongo.db.energy_usage.aggregate([
            {
                "$group": {
                    "_id": None,
                    "total_saved": {
                        "$sum": {
                            "$multiply": ["$usage", 0.2]  # Assume 20% savings from baseline
                        }
                    }
                }
            }
        ]).next()["total_saved"] if mongo.db.energy_usage.count_documents({}) > 0 else 0

        # Calculate total carbon reduced (0.5 kg CO2 per kWh saved)
        carbon_reduced = energy_saved * 0.5

        # Get eco tip of the day
        eco_tips = [
            "Switch to LED bulbs to save up to 80% on lighting energy costs",
            "Unplug electronics when not in use to avoid phantom energy consumption",
            "Use natural light during the day to reduce electricity usage",
            "Set your thermostat a few degrees lower in winter and higher in summer",
            "Regular maintenance of appliances ensures optimal energy efficiency",
            "Use cold water for laundry when possible to save energy",
            "Install a programmable thermostat to optimize heating and cooling",
            "Use power strips to easily turn off multiple devices at once",
            "Choose Energy Star certified appliances for better efficiency",
            "Clean or replace air filters regularly for better HVAC efficiency"
        ]
        current_day = get_current_time().timetuple().tm_yday
        eco_tip = eco_tips[current_day % len(eco_tips)]

        stats = {
            'total_users': total_users,
            'energy_saved': round(energy_saved, 2),
            'carbon_reduced': round(carbon_reduced, 2)
        }

        return render_template('index.html',
                               stats=stats,
                               eco_tip=eco_tip)

    except Exception as e:
        print(f"Error in index route: {str(e)}")
        # Return basic template with default values if there's an error
        return render_template('index.html',
                               stats={'total_users': 0, 'energy_saved': 0, 'carbon_reduced': 0},
                               eco_tip="Start tracking your energy usage today to make a difference!")


@app.route('/waste_segregation')
@login_required
def waste_segregation_page():
    return render_template('waste_segregation.html')


@app.route('/food_waste')
@login_required
def food_waste_page():
    try:
        # Get stats from database
        stats = mongo.db.impact_stats.find_one({"_id": "global"}) or {
            "total_food_saved": 0,
            "meals_provided": 0,
            "co2_saved": 0
        }

        # Get available food items
        items = list(mongo.db.food_items.find({"status": "available"}))

        return render_template('food_waste.html', stats=stats, items=items)
    except Exception as e:
        print(f"Error in food_waste_page: {str(e)}")
        return render_template('food_waste.html',
                               stats={"total_food_saved": 0, "meals_provided": 0, "co2_saved": 0},
                               items=[])


@app.route('/impact_dashboard')
@login_required
def impact_dashboard_page():
    try:
        # Get impact stats
        stats = mongo.db.impact_stats.find_one({"_id": "global"}) or {}

        # Get recent activities
        activities = list(mongo.db.carbon_activities.find().sort('timestamp', -1).limit(5))

        return render_template('impact_dashboard.html', stats=stats, activities=activities)
    except Exception as e:
        print(f"Error in impact_dashboard: {str(e)}")
        return render_template('impact_dashboard.html', stats={}, activities=[])


@app.route('/carbon_tracker')
# @login_required
def carbon_tracker():
    try:
        # Get user's carbon tracking data
        user_data = mongo.db.users.find_one({'_id': current_user.get_id()})

        # Get user's recent activities
        activities = list(mongo.db.carbon_activities.find(
            {'user_id': current_user.get_id()}
        ).sort('timestamp', -1).limit(10))

        # Calculate statistics
        total_impact = user_data.get('total_carbon_impact', 0) if user_data else 0
        activities_count = user_data.get('activities_logged', 0) if user_data else 0

        # Get global stats for comparison
        global_stats = mongo.db.impact_stats.find_one({'_id': 'global'}) or {
            'total_carbon_tracked': 0,
            'total_activities': 0,
            'average_impact': 0
        }

        return render_template('carbon_tracker.html',
                               activities=activities,
                               total_impact=total_impact,
                               activities_count=activities_count,
                               global_stats=global_stats)
    except Exception as e:
        print(f"Error in carbon_tracker route: {str(e)}")
        return render_template('carbon_tracker.html',
                               activities=[],
                               total_impact=0,
                               activities_count=0,
                               global_stats={'total_carbon_tracked': 0, 'total_activities': 0, 'average_impact': 0})


# Energy monitoring routes
@app.route('/get_usage_data')
@login_required
def get_usage_data():
    try:
        user_id = current_user.get_id()

        # Get recent readings
        recent_readings = list(mongo.db.energy_usage.find(
            {'user_id': user_id},
            {'_id': 0}
        ).sort('timestamp', -1).limit(10))

        # Process readings for display
        IST = pytz.timezone('Asia/Kolkata')
        for reading in recent_readings:
            # Convert timestamp to IST
            timestamp = reading['timestamp']
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            ist_time = timestamp.astimezone(IST)
            reading['timestamp'] = ist_time.strftime('%I:%M %p')

            if 'cost' not in reading:
                reading['cost'] = float(reading['usage']) * 0.12

            # Add trend
            if 'trend' not in reading:
                reading['trend'] = 'up' if float(reading['usage']) > 1.0 else 'down'

        # Get hourly data for chart
        start_time = datetime.now(IST) - timedelta(hours=24)
        hourly_data = list(mongo.db.energy_usage.find({
            'user_id': user_id,
            'timestamp': {'$gte': start_time}
        }).sort('timestamp', 1))

        # Process chart data with proper timezone
        chart_data = {
            'labels': [],
            'values': []
        }

        if hourly_data:
            # Convert timestamps to IST and format
            chart_data['labels'] = []
            chart_data['values'] = []

            for reading in hourly_data:
                timestamp = reading['timestamp']
                if timestamp.tzinfo is None:
                    timestamp = pytz.UTC.localize(timestamp)
                ist_time = timestamp.astimezone(IST)
                chart_data['labels'].append(ist_time.strftime('%I:%M %p'))
                chart_data['values'].append(float(reading['usage']))

        # Calculate metrics
        current_usage = float(recent_readings[0]['usage']) if recent_readings else 0
        total_cost = sum(float(r.get('cost', r['usage'] * 0.12)) for r in recent_readings)

        # Find peak hour
        if hourly_data:
            peak_reading = max(hourly_data, key=lambda x: float(x['usage']))
            timestamp = peak_reading['timestamp']
            if timestamp.tzinfo is None:
                timestamp = pytz.UTC.localize(timestamp)
            ist_time = timestamp.astimezone(IST)
            peak_hour = ist_time.strftime('%I:%M %p')
        else:
            peak_hour = 'N/A'

        # Calculate predicted usage
        predicted_usage = sum(float(r['usage']) for r in recent_readings) / len(
            recent_readings) if recent_readings else 0

        metrics = {
            'current_usage': current_usage,
            'total_cost': total_cost,
            'peak_hour': peak_hour,
            'predicted_usage': predicted_usage
        }

        return jsonify({
            'recent_readings': recent_readings,
            'chart_data': chart_data,
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error in get_usage_data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/add_usage', methods=['POST'])
@login_required
def add_energy_usage():
    try:
        data = request.get_json()
        # Validate required fields
        required_fields = ['appliance', 'usage', 'duration']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'})

        # Validate numeric fields
        usage = float(data['usage'])
        duration = float(data['duration'])
        if usage <= 0 or duration <= 0:
            return jsonify({'success': False, 'error': 'Usage and duration must be positive numbers'})

        # Create reading document
        reading = {
            'user_id': current_user.get_id(),
            'timestamp': get_current_time(),
            'appliance': data['appliance'],
            'usage': usage,
            'duration': duration,
            'usage_per_hour': usage / duration if duration > 0 else 0
        }

        # Insert reading
        result = execute_mongo_operation(mongo.db.energy_usage.insert_one, reading)

        if not result:
            return jsonify({'success': False, 'error': 'Failed to save reading'})

        # Update user's total usage
        execute_mongo_operation(mongo.db.users.update_one,
                                {'_id': current_user.get_id()},
                                {'$inc': {'total_energy_usage': usage, 'readings_count': 1}},
                                upsert=True
                                )

        # Calculate trend
        previous_readings = list(execute_mongo_operation(mongo.db.energy_usage.find, {
            'user_id': current_user.get_id(),
            'appliance': data['appliance']
        }).sort('timestamp', -1).limit(5))

        trend = ((usage - (sum(r['usage'] for r in previous_readings) / len(previous_readings))) / (
                    sum(r['usage'] for r in previous_readings) / len(
                previous_readings)) * 100) if previous_readings else 0

        return jsonify({
            'success': True,
            'message': 'Reading added successfully',
            'reading_id': str(result.inserted_id),
            'trend': round(trend, 1)
        })

    except Exception as e:
        print(f"Error adding usage reading: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_usage')
@login_required
def get_usage():
    try:
        user_id = current_user.get_id()
        # Get last 24 hours of data filtered by user_id
        start_time = datetime.now(IST) - timedelta(hours=24)
        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': start_time},
                    'user_id': user_id
                }
            },
            {
                '$group': {
                    '_id': '$appliance',
                    'total_usage': {'$sum': '$usage'},
                    'count': {'$sum': 1}
                }
            }
        ]

        results = list(mongo.db.energy_usage.aggregate(pipeline))
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_summary')
@login_required
def get_summary():
    try:
        user_id = current_user.get_id()
        # Get today's data filtered by user_id
        today_start = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)

        pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': today_start},
                    'user_id': user_id
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_usage': {'$sum': '$usage'},
                    'total_cost': {'$sum': '$cost'},
                    'readings': {'$push': {
                        'appliance': '$appliance',
                        'usage': '$usage',
                        'timestamp': '$timestamp'
                    }}
                }
            }
        ]

        result = list(mongo.db.energy_usage.aggregate(pipeline))

        if not result:
            return jsonify({
                'total_usage': 0,
                'total_cost': 0,
                'peak_hour': None,
                'readings': []
            })

        data = result[0]
        readings = data.get('readings', [])

        # Calculate peak hour
        hour_usage = {}
        for reading in readings:
            hour = reading['timestamp'].hour
            hour_usage[hour] = hour_usage.get(hour, 0) + reading['usage']

        peak_hour = max(hour_usage.items(), key=lambda x: x[1])[0] if hour_usage else None

        return jsonify({
            'total_usage': data['total_usage'],
            'total_cost': data['total_cost'],
            'peak_hour': peak_hour,
            'readings': readings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_appliance_ranges')
@login_required
def get_appliance_ranges():
    """Return the typical ranges for all appliances"""
    appliances_with_icons = {
        'AC': {'icon': 'snowflake', 'ranges': APPLIANCE_POWER['AC']},
        'Refrigerator': {'icon': 'box', 'ranges': APPLIANCE_POWER['Refrigerator']},
        'Washing Machine': {'icon': 'washer', 'ranges': APPLIANCE_POWER['Washing Machine']},
        'TV': {'icon': 'tv', 'ranges': APPLIANCE_POWER['TV']},
        'Lights': {'icon': 'lightbulb', 'ranges': APPLIANCE_POWER['Lights']},
        'Computer': {'icon': 'laptop', 'ranges': APPLIANCE_POWER['Computer']},
        'Microwave': {'icon': 'microwave', 'ranges': APPLIANCE_POWER['Microwave']},
        'Water Heater': {'icon': 'hot-tub', 'ranges': APPLIANCE_POWER['Water Heater']},
        'Fan': {'icon': 'fan', 'ranges': APPLIANCE_POWER['Fan']},
        'Other': {'icon': 'plug', 'ranges': APPLIANCE_POWER['Other']}
    }
    return jsonify(appliances_with_icons)


def get_appliance_recommendations(usage_data):
    """Generate AI-driven recommendations based on usage patterns"""
    recommendations = []
    appliance_totals = {}

    # Calculate total usage per appliance
    for reading in usage_data:
        app = reading['appliance']
        if app not in appliance_totals:
            appliance_totals[app] = {
                'total': 0,
                'count': 0,
                'peak_usage': 0,
                'peak_time': None
            }

        appliance_totals[app]['total'] += reading['usage']
        appliance_totals[app]['count'] += 1

        if reading['usage'] > appliance_totals[app]['peak_usage']:
            appliance_totals[app]['peak_usage'] = reading['usage']
            appliance_totals[app]['peak_time'] = reading['timestamp']

    # Generate recommendations based on patterns
    for app, data in appliance_totals.items():
        avg_usage = data['total'] / data['count']
        typical = APPLIANCE_POWER[app]['typical']

        if avg_usage > typical * 1.2:  # Using 20% above typical as threshold
            if app == 'AC':
                recommendations.append({
                    'appliance': app,
                    'severity': 'high',
                    'tip': 'Consider setting AC temperature 1-2 degrees higher to save energy',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })
            elif app == 'Refrigerator':
                recommendations.append({
                    'appliance': app,
                    'severity': 'medium',
                    'tip': 'Check refrigerator door seal and avoid frequent door opening',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })
            elif app == 'Lights':
                recommendations.append({
                    'appliance': app,
                    'severity': 'low',
                    'tip': 'Consider switching to LED bulbs and utilizing natural light',
                    'saving_potential': f"{round((avg_usage - typical) * 0.12 * 24 * 30, 2)} per month"
                })

    return recommendations


def calculate_savings_potential(current_usage, recommendations):
    """Calculate potential savings from implementing recommendations"""
    monthly_savings = 0
    yearly_savings = 0

    for rec in recommendations:
        saving = float(rec['saving_potential'].split()[0])
        monthly_savings += saving

    yearly_savings = monthly_savings * 12

    return {
        'monthly': round(monthly_savings, 2),
        'yearly': round(yearly_savings, 2),
        'co2_reduction': round(yearly_savings * 0.85, 2)  # kg of CO2 per kWh saved
    }


@app.route('/api/food/items', methods=['POST'])
@login_required
def add_food_item():
    try:
        data = request.json
        item = {
            "type": data["type"],
            "quantity": float(data["quantity"]),
            "expiry": datetime.fromisoformat(data["expiry"]),
            "storage": data["storage"],
            "user_id": current_user.get_id(),  # Track who added the food
            "status": "available",
            "added_at": datetime.utcnow()  # Use UTC time for consistency
        }
        result = food_items.insert_one(item)

        # Do NOT update stats here (score should only increase when claimed)
        return jsonify({
            "success": True,
            "id": str(result.inserted_id),
            "message": "Food item added successfully"
        })
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid value: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to add food item: {str(e)}"}), 500


@app.route('/api/food/items', methods=['GET'])
@login_required
def get_food_items():
    try:
        # Get all available food items (not just the current user's)
        items = list(food_items.find({
            "status": "available",  # Only show available items
            "expiry": {"$gte": datetime.utcnow()}  # Only show non-expired items
        }).sort("expiry", 1))

        # Process items for display
        for item in items:
            item['_id'] = str(item['_id'])
            if isinstance(item['expiry'], datetime):
                item['expiry'] = item['expiry'].isoformat()
            if isinstance(item['added_at'], datetime):
                item['added_at'] = item['added_at'].isoformat()

        return jsonify(items)
    except Exception as e:
        return jsonify({"error": f"Failed to get food items: {str(e)}"}), 500


@app.route('/api/food/stats')
@login_required
def get_food_stats():
    try:
        user_id = current_user.get_id()
        stats = waste_stats.find_one({'user_id': user_id}) or {
            'total_food_saved': 0,
            'meals_provided': 0,
            'co2_saved': 0
        }

        # Remove MongoDB _id field
        if '_id' in stats:
            del stats['_id']

        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"Failed to get food stats: {str(e)}"}), 500


@app.route('/api/impact/food/stats')
@login_required
def get_impact_food_stats():
    try:
        # Get global stats from database
        stats = mongo.db.impact_stats.find_one({"_id": "global"}) or {
            "total_food_saved": 0,
            "meals_provided": 0,
            "co2_saved": 0
        }

        # Remove MongoDB _id field
        if '_id' in stats:
            del stats['_id']

        return jsonify(stats)
    except Exception as e:
        print(f"Error getting impact food stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/food/items/<item_id>/claim', methods=['POST'])
@login_required
def claim_food_item(item_id):
    try:
        user_id = current_user.get_id()

        # Find the food item
        item = food_items.find_one({
            '_id': ObjectId(item_id),
            'status': 'available'  # Only allow claiming available items
        })
        if not item:
            return jsonify({"error": "Item not found or already claimed"}), 404

        # Update the item status to "claimed"
        food_items.update_one(
            {'_id': ObjectId(item_id)},
            {
                '$set': {
                    'status': 'claimed',
                    'claimed_at': datetime.utcnow(),
                    'claimed_by': user_id
                }
            }
        )

        # Update global stats (increment score only when claimed)
        mongo.db.impact_stats.update_one(
            {"_id": "global"},
            {
                '$inc': {
                    "total_food_saved": item["quantity"],
                    "meals_provided": item["quantity"] * 2,  # Assuming 0.5kg per meal
                    "co2_saved": item["quantity"] * 2.5  # 2.5kg CO2 per kg food saved
                }
            },
            upsert=True
        )

        # Update user-specific stats
        waste_stats.update_one(
            {'user_id': user_id},
            {
                '$inc': {
                    'total_food_saved': item["quantity"],
                    'meals_provided': item["quantity"] * 2,
                    'co2_saved': item["quantity"] * 2.5
                },
                '$set': {
                    'last_activity': datetime.utcnow()
                }
            },
            upsert=True
        )

        return jsonify({
            "success": True,
            "message": "Item claimed successfully"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to claim item: {str(e)}"}), 500


@app.route('/get_waste_data')
# @login_required
def get_waste_data():
    try:
        user_id = current_user.get_id()
        # Get recent records filtered by user_id
        recent_records = list(waste_stats.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("timestamp", -1).limit(10))

        # Calculate totals
        pipeline = [
            {
                "$match": {
                    "user_id": user_id
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_recycled": {"$sum": "$quantity"},
                    "total_impact": {"$sum": "$impact"}
                }
            }
        ]

        totals = list(waste_stats.aggregate(pipeline))
        total_recycled = totals[0]["total_recycled"] if totals else 0
        total_impact = totals[0]["total_impact"] if totals else 0

        return jsonify({
            "recent_records": recent_records,
            "total_recycled": total_recycled,
            "total_impact": total_impact
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/add_waste', methods=['POST'])
@login_required
def add_waste():
    try:
        data = request.json
        waste_entry = {
            "user_id": current_user.get_id(),
            "waste_type": data["waste_type"],
            "quantity": float(data["quantity"]),
            "timestamp": datetime.now(IST),
            "impact": float(data["quantity"]) * 2.5  # Impact calculation
        }

        waste_stats.insert_one(waste_entry)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/carbon/activities', methods=['GET'])
@login_required
def get_carbon_activities():
    try:
        user_id = current_user.get_id()
        activities = list(carbon_activities.find(
            {"user_id": user_id},
            {"_id": 0}
        ).sort("timestamp", -1))
        return jsonify(activities)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/log_activity', methods=['POST'])
@login_required
def log_activity():
    try:
        print("Received activity log request")
        data = request.get_json()
        print(f"Activity data: {data}")

        # Validate required fields
        required_fields = ['activityType', 'specificActivity', 'quantity', 'unit']
        for field in required_fields:
            if field not in data:
                print(f"Missing field: {field}")
                return jsonify({'success': False, 'error': f'Missing required field: {field}'})

        # Validate quantity is a positive number
        try:
            quantity = float(data['quantity'])
            if quantity <= 0:
                return jsonify({'success': False, 'error': 'Quantity must be positive'})
        except (ValueError, TypeError):
            print(f"Invalid quantity value: {data['quantity']}")
            return jsonify({'success': False, 'error': 'Invalid quantity value'})

        # Calculate carbon impact
        carbon_impact = calculate_carbon_impact(
            data['activityType'],
            data['specificActivity'],
            quantity,
            data['unit']
        )

        print(f"Calculated carbon impact: {carbon_impact}")

        # Create activity document
        activity = {
            'user_id': current_user.get_id(),
            'timestamp': get_current_time(),
            'activity_type': data['activityType'],
            'specific_activity': data['specificActivity'],
            'quantity': quantity,
            'unit': data['unit'],
            'carbon_impact': carbon_impact
        }

        # Insert activity into database
        result = mongo.db.carbon_activities.insert_one(activity)

        if not result.inserted_id:
            print("Failed to insert activity")
            return jsonify({'success': False, 'error': 'Failed to save activity'})

        # Update user's total impact and activity count
        user_update = mongo.db.users.update_one(
            {'_id': current_user.get_id()},
            {
                '$inc': {
                    'total_carbon_impact': carbon_impact,
                    'activities_logged': 1
                }
            },
            upsert=True
        )

        if not user_update.acknowledged:
            print("Failed to update user stats")
            return jsonify({'success': False, 'error': 'Failed to update user statistics'})

        # Update global impact stats
        global_update = mongo.db.impact_stats.update_one(
            {'_id': 'global'},
            {
                '$inc': {
                    'total_carbon_tracked': carbon_impact,
                    'total_activities': 1
                }
            },
            upsert=True
        )

        if not global_update.acknowledged:
            print("Failed to update global stats")
            return jsonify({'success': False, 'error': 'Failed to update global statistics'})

        print("Activity logged successfully")
        return jsonify({
            'success': True,
            'message': 'Activity logged successfully',
            'carbon_impact': carbon_impact
        })

    except Exception as e:
        print(f"Error in log_activity: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def calculate_carbon_impact(activity_type, specific_activity, quantity, unit):
    """Calculate carbon impact in kg CO2e based on activity"""
    try:
        print(f"Calculating impact for: {activity_type} - {specific_activity} - {quantity} {unit}")

        # Convert specific_activity to lookup key
        activity_key = specific_activity.lower().replace('_', '')

        # Base factors in kg CO2e per unit
        impact_factors = {
            'transport': {
                'cartravel': 0.120,  # per km
                'bustravel': 0.040,  # per km
                'traintravel': 0.020,  # per km
                'airtravel': 0.150,  # per km
                'bicycle': 0.000  # per km
            },
            'energy': {
                'electricityusage': 0.200,  # per kWh
                'naturalgas': 0.100,  # per kWh
                'heatingoil': 0.250  # per gallon
            },
            'food': {
                'meatconsumption': 2.000,  # per kg
                'dairyproducts': 1.000,  # per kg
                'plantbasedmeals': 0.200  # per kg
            },
            'waste': {
                'landfillwaste': 0.300,  # per kg
                'recycling': 0.050,  # per kg
                'composting': 0.020  # per kg
            }
        }

        print(f"Activity key: {activity_key}")
        print(f"Impact factors available: {impact_factors[activity_type]}")

        # Get the impact factor
        if activity_type not in impact_factors:
            print(f"Unknown activity type: {activity_type}")
            return 0

        if activity_key not in impact_factors[activity_type]:
            print(f"Unknown specific activity: {activity_key}")
            return 0

        # Get base impact factor
        impact_factor = impact_factors[activity_type][activity_key]
        print(f"Base impact factor: {impact_factor}")

        # Convert quantity based on unit
        quantity = abs(float(quantity))  # Ensure positive quantity

        if unit == 'g' and activity_type in ['food', 'waste']:
            print(f"Converting {quantity}g to kg")
            quantity = quantity / 1000  # Convert g to kg
        elif unit == 'miles' and activity_type == 'transport':
            print(f"Converting {quantity} miles to km")
            quantity = quantity * 1.60934  # Convert miles to km

        print(f"Final quantity after conversion: {quantity}")

        # Calculate impact
        impact = impact_factor * quantity
        print(f"Raw impact calculation: {impact_factor} * {quantity} = {impact}")

        # Cap maximum impact per activity
        max_impact = 1000  # Maximum 1000 kg CO2e per single activity
        if impact > max_impact:
            print(f"Impact exceeded maximum ({impact} > {max_impact}), capping at {max_impact}")
            impact = max_impact

        final_impact = round(max(0, impact), 3)  # Ensure non-negative and round to 3 decimal places
        print(f"Final impact: {final_impact}")
        return final_impact

    except Exception as e:
        print(f"Error calculating carbon impact: {str(e)}")
        return 0  # Return 0 if there's any error


@app.route('/api/get_carbon_stats')
@login_required
def get_carbon_stats():
    try:
        print("Getting carbon stats")
        user_id = current_user.get_id()
        print(f"User ID: {user_id}")

        # Get current month's activities
        current_month = datetime.now(IST).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        next_month = (current_month + timedelta(days=32)).replace(day=1)

        print(f"Fetching activities from {current_month} to {next_month}")

        # Get current month's personal emissions using aggregation
        personal_pipeline = [
            {
                '$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': current_month, '$lt': next_month}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_emissions': {'$sum': '$carbon_impact'}
                }
            }
        ]

        personal_result = list(mongo.db.carbon_activities.aggregate(personal_pipeline))
        personal_emissions = personal_result[0]['total_emissions'] if personal_result else 0
        print(f"Personal emissions: {personal_emissions}")

        # Get last month's emissions using aggregation
        last_month = current_month - timedelta(days=1)
        last_month = last_month.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        last_month_pipeline = [
            {
                '$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': last_month, '$lt': current_month}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_emissions': {'$sum': '$carbon_impact'}
                }
            }
        ]

        last_month_result = list(mongo.db.carbon_activities.aggregate(last_month_pipeline))
        last_month_emissions = last_month_result[0]['total_emissions'] if last_month_result else 0
        print(f"Last month emissions: {last_month_emissions}")

        # Calculate reduction percentage
        if last_month_emissions > 0:
            reduction_percentage = ((last_month_emissions - personal_emissions) / last_month_emissions) * 100
        else:
            reduction_percentage = 0
        print(f"Reduction percentage: {reduction_percentage}")

        # Get community stats using aggregation
        community_pipeline = [
            {
                '$match': {
                    'timestamp': {'$gte': current_month, '$lt': next_month}
                }
            },
            {
                '$group': {
                    '_id': '$user_id',
                    'total_emissions': {'$sum': '$carbon_impact'},
                    'activity_count': {'$sum': 1}
                }
            },
            {
                '$sort': {'total_emissions': 1}
            }
        ]

        community_results = list(mongo.db.carbon_activities.aggregate(community_pipeline))
        print(f"Community results: {community_results}")

        # Calculate community stats
        active_users = len(community_results)
        community_emissions = sum(result['total_emissions'] for result in community_results)

        # Calculate user rank (1-based index)
        user_rank = next((i + 1 for i, r in enumerate(community_results) if r['_id'] == user_id), active_users)

        print(
            f"Stats calculated: Active users: {active_users}, Community emissions: {community_emissions}, User rank: {user_rank}")

        # Create index on timestamp and user_id if it doesn't exist
        mongo.db.carbon_activities.create_index([("timestamp", 1), ("user_id", 1)])

        response_data = {
            'personal_emissions': round(personal_emissions, 2),
            'reduction_percentage': round(reduction_percentage, 1),
            'community_emissions': round(community_emissions, 2),
            'active_users': max(active_users, 1),  # Ensure at least 1 active user
            'user_rank': min(user_rank, max(active_users, 1)),  # Ensure rank doesn't exceed total users
            'total_users': max(active_users, 1)  # Ensure at least 1 total user
        }

        print(f"Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"Error getting carbon stats: {str(e)}")
        # Return safe default values on error
        return jsonify({
            'personal_emissions': 0,
            'reduction_percentage': 0,
            'community_emissions': 0,
            'active_users': 1,
            'user_rank': 1,
            'total_users': 1
        })


@app.route('/api/get_leaderboard')
@login_required
def get_leaderboard():
    try:
        current_month = datetime.now(IST).replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Get all users and their emissions
        users = carbon_activities.distinct('user_id')
        leaders = []

        for user_id in users:
            # Current month emissions
            current_emissions = sum(a['carbon_impact'] for a in carbon_activities.find({
                'user_id': user_id,
                'timestamp': {'$gte': current_month}
            }))

            # Last month emissions
            last_month = (current_month - timedelta(days=1)).replace(day=1)
            last_emissions = sum(a['carbon_impact'] for a in carbon_activities.find({
                'user_id': user_id,
                'timestamp': {'$gte': last_month, '$lt': current_month}
            }))

            # Calculate reduction
            reduction = 0
            if last_emissions > 0:
                reduction = ((last_emissions - current_emissions) / last_emissions) * 100
                reduction = max(0, reduction)

            leaders.append({
                'name': f'User {user_id[-4:]}',  # Using last 4 chars of user_id for demo
                'emissions': round(current_emissions, 2),
                'reduction': round(reduction, 1)
            })

        # Sort by emissions (lower is better)
        leaders.sort(key=lambda x: x['emissions'])

        return jsonify({'leaders': leaders[:10]})  # Return top 10
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/get_personalized_tips')
@login_required
def get_personalized_tips():
    try:
        user_id = current_user.get_id()

        # Get user's recent activities
        recent_activities = list(mongo.db.carbon_activities.find({
            'user_id': user_id
        }).sort('timestamp', -1).limit(10))

        # Analyze activities to generate personalized tips
        tips = []
        activity_types = set(a.get('activity_type', '') for a in recent_activities)

        # Add activity-specific tips
        if 'transport' in activity_types:
            tips.append({
                'category': 'info',
                'icon': 'fa-car',
                'title': 'Transportation Tip',
                'description': 'Consider carpooling or using public transport to reduce your carbon footprint.'
            })

        if 'energy' in activity_types:
            tips.append({
                'category': 'warning',
                'icon': 'fa-bolt',
                'title': 'Energy Saving Tip',
                'description': 'Switch to LED bulbs and turn off appliances when not in use.'
            })

        if 'food' in activity_types:
            tips.append({
                'category': 'success',
                'icon': 'fa-utensils',
                'title': 'Food Choice Tip',
                'description': 'Try incorporating more plant-based meals into your diet.'
            })

        if 'waste' in activity_types:
            tips.append({
                'category': 'primary',
                'icon': 'fa-recycle',
                'title': 'Waste Management Tip',
                'description': 'Start composting your food waste to reduce methane emissions.'
            })

        # Always add general tips if we have less than 3 tips
        general_tips = [
            {
                'category': 'secondary',
                'icon': 'fa-leaf',
                'title': 'Start Small',
                'description': 'Begin with simple changes like using reusable bags and water bottles.'
            },
            {
                'category': 'info',
                'icon': 'fa-lightbulb',
                'title': 'Energy Efficiency',
                'description': 'Use natural light when possible and switch to LED bulbs.'
            },
            {
                'category': 'success',
                'icon': 'fa-bicycle',
                'title': 'Green Transport',
                'description': 'Walk, cycle, or use public transport for short distances.'
            }
        ]

        # Add general tips until we have at least 3 tips
        while len(tips) < 3:
            tip = general_tips[len(tips)]
            if tip not in tips:
                tips.append(tip)

        return jsonify({'tips': tips})
    except Exception as e:
        print(f"Error getting tips: {str(e)}")
        # Return default tips on error
        default_tips = [
            {
                'category': 'secondary',
                'icon': 'fa-leaf',
                'title': 'Start Your Journey',
                'description': 'Track your daily activities to see your carbon impact.'
            },
            {
                'category': 'info',
                'icon': 'fa-lightbulb',
                'title': 'Quick Win',
                'description': 'Turn off lights when leaving rooms.'
            },
            {
                'category': 'success',
                'icon': 'fa-recycle',
                'title': 'Reduce & Reuse',
                'description': 'Start with simple recycling and reusing everyday items.'
            }
        ]
        return jsonify({'tips': default_tips})


# Environmental Impact Dashboard Routes
@app.route('/user_guide')
def user_guide():
    return render_template('user_guide.html')


@app.route('/get_dashboard_data')
def get_dashboard_data():
    try:
        # Get current user's data
        carbon_data = list(mongo.db.carbon_activities.find({}).sort('timestamp', -1).limit(30))
        waste_data = list(mongo.db.waste_stats.find({}).sort('timestamp', -1).limit(30))

        # Calculate totals with proper handling of None values
        carbon_offset = sum(float(activity.get('carbon_impact', 0)) for activity in carbon_data if activity.get('carbon_impact') is not None)
        waste_reduction = sum(float(stat.get('items_recycled', 0)) for stat in waste_data if stat.get('items_recycled') is not None)

        # Calculate progress percentages with proper handling of division by zero
        carbon_progress = min(100, (carbon_offset / 100) * 100) if carbon_offset > 0 else 0
        waste_progress = min(100, (waste_reduction / 50) * 100) if waste_reduction > 0 else 0

        # Get recent activities for timeline
        recent_activities = list(mongo.db.carbon_activities.find({})
                                 .sort('timestamp', -1)
                                 .limit(5))

        timeline = []
        for activity in recent_activities:
            timeline.append({
                'title': activity.get('activity_type', 'Activity').title(),
                'description': f"Impact: {abs(float(activity.get('carbon_impact', 0)))} kg CO2",
                'date': activity.get('timestamp', datetime.now()).strftime('%Y-%m-%d')
            })

        # Get achievements
        achievements = [
            {
                'name': 'Green Thumb',
                'icon': 'fa-seedling',
                'unlocked': carbon_offset > 50
            },
            {
                'name': 'Energy Saver',
                'icon': 'fa-bolt',
                'unlocked': carbon_offset > 100
            },
            {
                'name': 'Recycler',
                'icon': 'fa-recycle',
                'unlocked': waste_reduction > 30
            }
        ]

        # Get active challenges
        challenges = [
            {
                'title': 'Zero Waste Week',
                'description': 'Minimize your waste production for 7 days',
                'status': 'active',
                'progress': min(100, (waste_reduction / 20) * 100) if waste_reduction > 0 else 0
            },
            {
                'title': 'Energy Saver',
                'description': 'Reduce energy consumption by 20%',
                'status': 'active',
                'progress': min(100, (carbon_offset / 50) * 100) if carbon_offset > 0 else 0
            }
        ]

        # Calculate eco score (0-100)
        eco_score = min(100, int((carbon_offset + waste_reduction) / 2))

        # Get insights based on data
        insights = [
            {
                'title': 'Carbon Reduction',
                'description': f'You have offset {carbon_offset:.1f} kg of CO2. Keep up the good work!'
            },
            {
                'title': 'Waste Management',
                'description': f'You have recycled {waste_reduction} items. This helps reduce landfill waste.'
            }
        ]

        # Get community stats
        total_users = mongo.db.carbon_activities.distinct('user_id')
        user_rank = random.randint(1, max(len(total_users), 1))

        return jsonify({
            'eco_score': eco_score,
            'carbon_offset': round(carbon_offset, 1),
            'waste_reduction': int(waste_reduction),
            'progress': {
                'carbonProgress': round(carbon_progress, 1),
                'wasteProgress': round(waste_progress, 1)
            },
            'timeline': timeline,
            'insights': insights,
            'achievements': achievements,
            'challenges': challenges,
            'community': {
                'rank': user_rank,
                'rank_percentile': round((1 - (user_rank / max(len(total_users), 1))) * 100, 1),
                'contribution': round(random.uniform(60, 90), 1)
            }
        })
    except Exception as e:
        print(f"Dashboard error: {e}")
        return jsonify({
            'eco_score': 0,
            'carbon_offset': 0,
            'waste_reduction': 0,
            'progress': {'carbonProgress': 0, 'wasteProgress': 0},
            'timeline': [],
            'insights': [],
            'achievements': [],
            'challenges': [],
            'community': {'rank': 0, 'rank_percentile': 0, 'contribution': 0}
        }), 500

@app.route('/get_user_stats')
@login_required
def get_user_stats():
    """Get user stats from waste_stats collection"""
    try:
        # Get stats from database
        stats = mongo.db.waste_stats.find_one({'user_id': current_user.get_id()}) or {
            'items_recycled': 0,
            'eco_points': 0,
            'streak': 0,
            'total_recyclable': 0,
            'total_compostable': 0,
            'total_waste': 0,
            'total_items': 0
        }

        # Remove MongoDB _id
        if '_id' in stats:
            del stats['_id']

        return jsonify(stats)
    except Exception as e:
        print(f"Error getting user stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_educational_fact')
@login_required
def get_educational_fact():
    # List of educational facts about waste management
    facts = [
        "Recycling one aluminum can saves enough energy to run a TV for three hours.",
        "The average person generates over 4 pounds of trash every day.",
        "Glass can be recycled endlessly without losing quality.",
        "Plastic bags take 10-1000 years to decompose.",
        "About 75% of waste is recyclable, but we only recycle about 30%.",
        "Composting can reduce your household waste by 30%.",
        "E-waste represents 2% of trash in landfills but 70% of toxic waste.",
        "Every ton of paper recycled saves 17 trees."
    ]
    return jsonify({'fact': random.choice(facts)})


@app.route('/get_impact_data')
@login_required
def get_impact_data():
    try:
        # Get user's impact data
        user_id = current_user.get_id()
        impact_data = mongo.db.impact_data.find_one({'user_id': user_id}) or {
            'water_saved': 0,
            'water_saved_percentage': 0,
            'energy_saved': 0,
            'energy_saved_percentage': 0,
            'co2_reduced': 0,
            'co2_reduced_percentage': 0,
            'equivalents': [
                {
                    'icon': 'fa-car',
                    'color': 'primary',
                    'value': '0 km',
                    'description': 'Car trips avoided'
                },
                {
                    'icon': 'fa-tree',
                    'color': 'success',
                    'value': '0 trees',
                    'description': 'Equivalent trees planted'
                },
                {
                    'icon': 'fa-lightbulb',
                    'color': 'warning',
                    'value': '0 hours',
                    'description': 'LED bulb runtime'
                }
            ]
        }

        # Remove MongoDB _id field
        if '_id' in impact_data:
            del impact_data['_id']

        return jsonify(impact_data)
    except Exception as e:
        print(f"Error getting impact data: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/food/stats')
@login_required
def get_dashboard_food_stats():
    try:
        # Get stats from database
        stats = mongo.db.impact_stats.find_one({"_id": "global"}) or {
            "total_food_saved": 0,
            "meals_provided": 0,
            "co2_saved": 0
        }

        # Remove MongoDB _id field
        if '_id' in stats:
            del stats['_id']

        return jsonify(stats)
    except Exception as e:
        print(f"Error getting dashboard food stats: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/get_energy_tips')
@login_required
def get_energy_tips():
    try:
        user_id = current_user.get_id()

        # Get user's recent usage data
        start_time = datetime.now(IST) - timedelta(days=30)
        monthly_usage = list(mongo.db.energy_usage.find({
            'user_id': user_id,
            'timestamp': {'$gte': start_time}
        }))

        # Calculate total monthly usage and cost
        total_usage = sum(float(reading['usage']) for reading in monthly_usage)
        total_cost = sum(float(reading.get('cost', reading['usage'] * 0.12)) for reading in monthly_usage)

        # Calculate potential savings (assume 20% reduction is achievable)
        monthly_savings = total_cost * 0.20
        yearly_savings = monthly_savings * 12

        # Calculate CO2 reduction (0.5 kg CO2 per kWh saved)
        potential_monthly_savings_kwh = total_usage * 0.20
        co2_reduction = potential_monthly_savings_kwh * 0.5 * 12  # yearly CO2 reduction

        # Get appliance-specific tips based on usage patterns
        appliance_usage = {}
        for reading in monthly_usage:
            appliance = reading['appliance']
            if appliance not in appliance_usage:
                appliance_usage[appliance] = 0
            appliance_usage[appliance] += float(reading['usage'])

        # Generate personalized tips based on usage patterns
        tips = []

        # Add general tips
        tips.extend([
            "Turn off lights when leaving rooms",
            "Use natural light during daytime",
            "Set AC temperature to 24°C for optimal efficiency"
        ])

        # Add appliance-specific tips
        if appliance_usage.get('Air Conditioner', 0) > 100:
            tips.append("Consider using fans along with AC to improve efficiency")
            tips.append("Clean AC filters regularly for better performance")

        if appliance_usage.get('Refrigerator', 0) > 50:
            tips.append("Keep refrigerator coils clean for better efficiency")
            tips.append("Maintain optimal temperature settings for your refrigerator")

        if appliance_usage.get('Washing Machine', 0) > 30:
            tips.append("Run full loads of laundry to save energy")
            tips.append("Use cold water when possible for washing clothes")

        # Limit to top 5 most relevant tips
        tips = tips[:5]

        return jsonify({
            'tips': tips,
            'savings': {
                'monthly': round(monthly_savings, 2),
                'yearly': round(yearly_savings, 2),
                'co2_reduction': round(co2_reduction, 1)
            }
        })
    except Exception as e:
        print(f"Error in get_energy_tips: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_waste', methods=['POST'])
def analyze_waste():
    """Analyze waste image using Gemini API"""
    try:
        print("\n=== Starting waste analysis ===")
        # Get image from request
        if 'image' not in request.files:
            print("No image file in request")
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']
        if not image_file.filename:
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        # Print debug information
        print(f"Received file: {image_file.filename}")
        print(f"File content type: {image_file.content_type}")

        # Read and process image
        try:
            # Read image bytes
            image_bytes = image_file.read()
            print(f"Read {len(image_bytes)} bytes from image file")

            # Create BytesIO object
            image_buffer = io.BytesIO(image_bytes)

            # Open image with PIL
            image = Image.open(image_buffer)
            print(f"Image opened successfully: format={image.format}, size={image.size}, mode={image.mode}")

            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                print(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')

            # Ensure image is not too large (max 4MB after processing)
            max_size = (1024, 1024)  # Maximum dimensions
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                print(f"Resizing image from {image.size} to max dimensions {max_size}")
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # Convert to JPEG format in memory
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='JPEG', quality=85)
            output_buffer.seek(0)
            processed_image = output_buffer.getvalue()
            print(f"Processed image size: {len(processed_image)} bytes")

            # Configure Gemini API
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

            # Create model instance
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Prepare prompt
            prompt = """Analyze this image and classify the waste item. Respond ONLY with a valid JSON object in this exact format, nothing else:
            {
                "item_type": "brief description of the item",
                "category": "one of: recyclable, compostable, or waste",
                "instructions": "brief disposal instructions",
                "impact": "brief environmental impact"
            }"""

            # Generate response
            print("\nSending request to Gemini API...")
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": processed_image}
            ])

            # Get the response text
            response_text = response.text.strip()
            print(f"Received response from Gemini API: {response_text}")

            try:
                # Find JSON object in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start == -1 or json_end == -1:
                    raise ValueError("No JSON object found in response")

                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Validate and clean the result
                required_fields = ["item_type", "category", "instructions", "impact"]
                for field in required_fields:
                    if field not in result or not result[field]:
                        result[field] = "Not specified"

                # Normalize category
                result["category"] = result["category"].lower().strip()
                if result["category"] not in ["recyclable", "compostable", "waste"]:
                    result["category"] = "waste"

                # Calculate points
                points = {
                    'recyclable': 15,
                    'compostable': 12,
                    'waste': 10
                }.get(result['category'], 10)

                # Update user stats
                mongo.db.waste_stats.update_one(
                    {'user_id': current_user.get_id()},
                    {
                        '$inc': {
                            'items_recycled': 1,
                            'eco_points': points,
                            f'total_{result["category"]}': 1,
                            'total_items': 1
                        },
                        '$set': {
                            'last_activity_date': datetime.now(IST)
                        }
                    },
                    upsert=True
                )

                # Add waste record
                waste_record = {
                    'user_id': current_user.get_id(),
                    'timestamp': datetime.now(IST),
                    'item_type': result['item_type'],
                    'category': result['category'],
                    'points_earned': points,
                    'environmental_impact': result['impact']
                }
                mongo.db.food_waste.insert_one(waste_record)

                # Get current stats
                user_stats = mongo.db.waste_stats.find_one({'user_id': current_user.get_id()}) or {
                    'items_recycled': 0,
                    'eco_points': 0,
                    'streak': 0
                }

                # Educational facts
                facts = [
                    "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
                    "Glass bottles can be recycled endlessly without quality degradation.",
                    "Plastic bags take 10-1000 years to decompose in landfills.",
                    "Composting food waste reduces methane emissions from landfills.",
                    "Recycling paper saves trees and reduces water pollution.",
                    "E-waste recycling recovers valuable metals and prevents toxic pollution."
                ]

                response_data = {
                    **result,
                    'stats': {
                        'items_recycled': user_stats['items_recycled'],
                        'eco_points': user_stats['eco_points'],
                        'streak': user_stats.get('streak', 0)
                    },
                    'points_earned': points,
                    'educational_fact': random.choice(facts)
                }

                print(f"Sending successful response: {response_data}")
                return jsonify(response_data)

            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                return jsonify({
                    'error': 'Failed to parse analysis results. Please try again.'
                }), 500

        except Exception as img_error:
            print(f"Image processing error: {str(img_error)}")
            return jsonify({
                'error': f'Failed to process image: {str(img_error)}'
            }), 400

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred. Please try again.'
        }), 500


@app.route('/api/add_usage', methods=['POST'])
# @login_required
def add_usage():
    try:
        data = request.get_json()
        print(f"Received usage data: {data}")

        # Validate required fields
        required_fields = ['appliance', 'usage', 'duration']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'})

        # Validate numeric fields
        try:
            usage = float(data['usage'])
            duration = float(data['duration'])
            if usage <= 0 or duration <= 0:
                return jsonify({'success': False, 'error': 'Usage and duration must be positive numbers'})
        except (ValueError, TypeError):
            return jsonify({'success': False, 'error': 'Invalid numeric values'})

        # Create reading document
        reading = {
            'user_id': current_user.get_id(),
            'timestamp': get_current_time(),
            'appliance': data['appliance'],
            'usage': usage,
            'duration': duration,
            'usage_per_hour': usage / duration if duration > 0 else 0
        }

        # Insert reading
        result = execute_mongo_operation(mongo.db.energy_usage.insert_one, reading)

        if not result:
            return jsonify({'success': False, 'error': 'Failed to save reading'})

        # Update user's total usage
        execute_mongo_operation(mongo.db.users.update_one,
                                {'_id': current_user.get_id()},
                                {'$inc': {'total_energy_usage': usage, 'readings_count': 1}},
                                upsert=True
                                )

        # Calculate trend
        previous_readings = list(execute_mongo_operation(mongo.db.energy_usage.find, {
            'user_id': current_user.get_id(),
            'appliance': data['appliance']
        }).sort('timestamp', -1).limit(5))

        trend = ((usage - (sum(r['usage'] for r in previous_readings) / len(previous_readings))) / (
                    sum(r['usage'] for r in previous_readings) / len(
                previous_readings)) * 100) if previous_readings else 0

        return jsonify({
            'success': True,
            'message': 'Reading added successfully',
            'reading_id': str(result.inserted_id),
            'trend': round(trend, 1)
        })

    except Exception as e:
        print(f"Error adding usage reading: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/get_energy_usage_data')
@login_required
def get_energy_usage_data():
    try:
        user_id = current_user.get_id()

        # Get recent readings with trend calculation
        recent_readings = []
        readings = list(mongo.db.energy_usage.find(
            {'user_id': user_id}
        ).sort('timestamp', -1).limit(10))

        for reading in readings:
            # Calculate trend for each reading
            previous_readings = list(mongo.db.energy_usage.find({
                'user_id': user_id,
                'appliance': reading['appliance'],
                'timestamp': {'$lt': reading['timestamp']}
            }).sort('timestamp', -1).limit(5))

            if previous_readings:
                avg_usage = sum(r['usage'] for r in previous_readings) / len(previous_readings)
                trend = ((reading['usage'] - avg_usage) / avg_usage * 100) if avg_usage > 0 else 0
            else:
                trend = 0

            recent_readings.append({
                'timestamp': reading['timestamp'],
                'appliance': reading['appliance'],
                'usage': reading['usage'],
                'duration': reading['duration'],
                'trend': round(trend, 1)
            })

        # Get chart data (last 60 minutes)
        sixty_mins_ago = get_current_time() - timedelta(minutes=60)
        chart_data = mongo.db.energy_usage.aggregate([
            {
                '$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': sixty_mins_ago}
                }
            },
            {
                '$group': {
                    '_id': {
                        '$dateToString': {
                            'format': '%H:%M',
                            'date': '$timestamp'
                        }
                    },
                    'total_usage': {'$sum': '$usage'}
                }
            },
            {'$sort': {'_id': 1}}
        ])

        chart_labels = []
        chart_values = []

        for item in chart_data:
            chart_labels.append(item['_id'])
            chart_values.append(round(item['total_usage'], 2))

        # Calculate current month's usage
        current_month = get_current_time().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_usage = mongo.db.energy_usage.aggregate([
            {
                '$match': {
                    'user_id': user_id,
                    'timestamp': {'$gte': current_month}
                }
            },
            {
                '$group': {
                    '_id': None,
                    'total_usage': {'$sum': '$usage'}
                }
            }
        ]).next()['total_usage'] if mongo.db.energy_usage.count_documents({'user_id': user_id}) > 0 else 0

        # Calculate daily average
        days_in_month = get_current_time().day
        daily_average = current_usage / days_in_month if days_in_month > 0 else 0

        # Calculate carbon footprint (0.5 kg CO2 per kWh)
        carbon_footprint = current_usage * 0.5

        return jsonify({
            'success': True,
            'current_usage': round(current_usage, 2),
            'daily_average': round(daily_average, 2),
            'carbon_footprint': round(carbon_footprint, 2),
            'recent_readings': recent_readings,
            'chart_data': {
                'labels': chart_labels,
                'values': chart_values
            }
        })

    except Exception as e:
        print(f"Error getting usage data: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/energy_monitor')
# @login_required
def energy_monitor():
    try:
        # Get user's energy data
        user_id = current_user.get_id()

        # Get recent readings
        recent_readings = list(mongo.db.energy_usage.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(10))

        # Calculate current usage and daily average
        current_month = get_current_time().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        current_usage = mongo.db.energy_usage.aggregate([
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": current_month}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_usage": {"$sum": "$usage"}
                }
            }
        ]).next()["total_usage"] if mongo.db.energy_usage.count_documents({"user_id": user_id}) > 0 else 0

        # Calculate daily average
        daily_average = current_usage / (get_current_time().day) if current_usage > 0 else 0

        # Calculate carbon footprint (rough estimate: 0.5 kg CO2 per kWh)
        carbon_footprint = current_usage * 0.5

        # Get appliance list for dropdown
        appliances = [
            'Refrigerator',
            'Air Conditioner',
            'Washing Machine',
            'Television',
            'Computer',
            'Water Heater',
            'Microwave',
            'Dishwasher',
            'Lighting',
            'Other'
        ]

        return render_template('energy_monitor.html',
                               current_usage=current_usage,
                               daily_average=daily_average,
                               carbon_footprint=carbon_footprint,
                               recent_readings=recent_readings,
                               appliances=appliances)

    except Exception as e:
        print(f"Error in energy_monitor route: {str(e)}")
        flash('Error loading energy monitor', 'error')
        return redirect(url_for('index'))

@app.route('/api/get_energy_dashboard_data')
@login_required
def get_energy_dashboard_data():
    try:
        # Get total users
        total_users = mongo.db.users.count_documents({})

        # Calculate total energy saved (assume 20% savings from baseline)
        total_energy = mongo.db.energy_usage.aggregate([
            {
                "$group": {
                    "_id": None,
                    "total_saved": {
                        "$sum": {
                            "$multiply": ["$usage", 0.2]  # 20% savings
                        }
                    }
                }
            }
        ]).next()['total_saved'] if mongo.db.energy_usage.count_documents({}) > 0 else 0

        # Calculate carbon reduction (0.5 kg CO2 per kWh saved)
        carbon_reduced = total_energy * 0.5

        # Get eco tip of the day
        eco_tips = [
            "Switch to LED bulbs to save up to 80% on lighting energy costs",
            "Unplug electronics when not in use to avoid phantom energy consumption",
            "Use natural light during the day to reduce electricity usage",
            "Set your thermostat a few degrees lower in winter and higher in summer",
            "Regular maintenance of appliances ensures optimal energy efficiency",
            "Use cold water for laundry when possible to save energy",
            "Install a programmable thermostat to optimize heating and cooling",
            "Use power strips to easily turn off multiple devices at once",
            "Choose Energy Star certified appliances for better efficiency",
            "Clean or replace air filters regularly for better HVAC efficiency"
        ]
        current_day = get_current_time().timetuple().tm_yday
        eco_tip = eco_tips[current_day % len(eco_tips)]

        return jsonify({
            'success': True,
            'total_users': total_users,
            'energy_saved': round(total_energy, 2),
            'carbon_reduced': round(carbon_reduced, 2),
            'eco_tip': eco_tip
        })

    except Exception as e:
        print(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'total_users': 0,

            'energy_saved': 0,
            'carbon_reduced': 0,
            'eco_tip': "Start tracking your energy usage today!"
        })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5676)
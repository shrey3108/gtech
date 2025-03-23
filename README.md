# EcoTrack+ - Smart Energy & Sustainability Platform

A comprehensive platform for energy monitoring, waste management, and sustainable living.

## Features

- Real-time energy usage monitoring
- Waste segregation tracking
- Food waste management
- Carbon footprint calculator
- Environmental impact dashboard
- AI-powered recommendations

## Tech Stack

- Python 3.10
- Flask
- MongoDB
- Chart.js
- Google Generative AI

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables in `.env`:
   ```
   MONGODB_URI=your_mongodb_uri
   GOOGLE_API_KEY=your_google_api_key
   SECRET_KEY=your_secret_key
   ```
5. Run the application:
   ```bash
   python app.py
   ```

## Deployment to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables:
   - `MONGODB_URI`: Your MongoDB Atlas connection string
   - `GOOGLE_API_KEY`: Your Google API key
   - `SECRET_KEY`: A secure random string

## Environment Variables

- `MONGODB_URI`: MongoDB connection string
- `GOOGLE_API_KEY`: Google API key for AI features
- `SECRET_KEY`: Flask secret key for session management

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── gunicorn.conf.py   # Gunicorn configuration
├── render.yaml        # Render deployment configuration
├── .env              # Environment variables (local development)
├── .gitignore        # Git ignore rules
└── templates/        # HTML templates
    ├── base.html
    ├── index.html
    ├── energy_monitor.html
    └── ...

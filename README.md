# Cutoff Prediction API

A Flask REST API for predicting cutoffs using machine learning.

## Features
- REST endpoint `/predict` for generating cutoff predictions.
- Powered by Flask, LightGBM, Scikit-learn, and Pandas.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/somesh123211/pd.git
   cd pd
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API:**
   ```bash
   python app.py
   ```

## API Usage

### `POST /predict`
**Request Body (JSON):**
```json
{
  "year": 2026,
  "university_id": "101",
  "programme": "CSE",
  "category": "General",
  "round": "1"
}
```

**Response (JSON):**
```json
{
  "status": "success",
  "predicted_cutoff": 123.45
}
```

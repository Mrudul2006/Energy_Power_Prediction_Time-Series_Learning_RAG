# EnergyCast: AI-Powered Grid Forecasting

EnergyCast is a state-of-the-art energy consumption forecasting platform. It bridges the gap between static machine learning models and real-time grid dynamics by introducing **Online Learning** and **Explainable AI (XAI)** into an interactive, highly visual dashboard.

![EnergyCast Dashboard](https://via.placeholder.com/1000x500?text=EnergyCast+Dashboard)

## 🌟 Key Features

* **Rolling Time-Series Forecasting**: View an interactive 24-hour rolling forecast of grid demand, continuously anchoring to system time.
* **Online Learning (Residual Corrector)**: A Ridge regression-based online learner tracks real-time forecast errors and corrects the base ensemble model on the fly. As new actual demand observations are submitted, the system adapts instantly to combat concept drift.
* **Explainable AI (XAI)**: Powered by Google's Gemini 2.0 Flash, the built-in explainer analyzes grid conditions, load shapes, and forecast anomalies, translating raw MW predictions into actionable natural language insights for grid operators.
* **Historical Backtesting**: A dedicated 2021 backtest module to visualize the long-term performance improvement (MAE/RMSE reductions) provided by the online learning corrector across hourly, daily, weekly, and monthly resolutions.
* **Dynamic UI Toggles**: Interactive, dependency-free SVG charts allowing you to seamlessly toggle visibility between Actual Demand, Base Model Predictions, and Online Corrected Predictions.

## 🛠 Tech Stack

### Frontend
* **Framework**: Next.js 15 (React 19)
* **Styling**: Tailwind CSS & Vanilla CSS (Custom Design System)
* **Charting**: Custom dependency-free responsive SVG charting system
* **Language**: TypeScript

### Backend
* **Framework**: FastAPI (Uvicorn)
* **Machine Learning**: Pandas, Scikit-Learn (Ridge), NumPy
* **Explainable AI**: Google Generative AI (Gemini 2.0 REST API)
* **State Management**: Pickle for persisting the online learner state

## 📂 Project Structure

```text
Energy_Conservation_ML/
├── backend/
│   ├── main.py              # FastAPI application and routing
│   ├── explainer.py         # Gemini API integration for XAI
│   ├── online_learner.py    # Ridge regression residual corrector
│   └── models/              # Pre-trained base models (e.g. MSST-Net, LSTM)
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main dashboard interface
│   │   ├── api/             # Next.js API proxies (/history, /forecast, /explain, /2021)
│   │   ├── layout.tsx       # Root HTML layout
│   │   └── globals.css      # Core design tokens and CSS resets
│   └── lib/
│       └── forecast.ts      # Client-side data normalization and forecasting types
├── cleaned_energy_data.csv  # Working dataset
└── README.md                # Project documentation
```

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* Node.js 18+

### 1. Backend Setup

Open a terminal in the root directory:

```bash
# Optional: Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (assuming a requirements.txt exists)
# pip install -r requirements.txt
pip install fastapi uvicorn pandas scikit-learn numpy requests

# Start the API server
python -m uvicorn backend.main:app --reload
```
The backend will run on `http://localhost:8000`.

### 2. Frontend Setup

Open a second terminal and navigate to the frontend directory:

```bash
cd frontend

# Install dependencies
npm install

# Start the Next.js development server
npm run dev
```
The frontend dashboard will be available at `http://localhost:3000`.

### 3. Environment Variables
If you intend to use the Explainable AI feature, ensure you have set up your API keys in the backend (e.g., inside `backend/explainer.py` or `.env`).

## 🧠 How the Architecture Works

1. **The Base Prediction**: The backend evaluates a pre-trained ensemble model to establish a baseline forecast for grid demand.
2. **The Online Correction**: If historical "actual" values are provided by the frontend, the `OnlineLearner` class fits a Ridge regression model against the recent residual errors. It then adjusts the base prediction, returning a highly accurate `predicted_online` value.
3. **The AI Explanation**: When the "Generate AI Insights" button is clicked, the frontend serializes the entire current forecast window and sends it to `/api/explain`. The backend parses this, builds a structured prompt describing grid peaks/troughs, and queries the LLM to return structured JSON insights.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

# Personalized AI Financial Advisory

## Overview
This project is a personalized AI-driven financial advisory application that provides users with tailored asset allocation recommendations based on their financial profile, goals, and risk tolerance. The application uses machine learning models to predict optimal asset allocation percentages and provides detailed explanations for the recommendations.

## Features
- **User Input Form**: Collects user details such as age, income, expenses, risk tolerance, financial goals, and investment experience.
- **Asset Allocation Prediction**: Uses a pre-trained machine learning model to recommend allocations for equity, debt, gold, and real estate.
- **Explanations**: Provides detailed explanations for the recommendations based on user inputs.
- **PDF Report Generation**: Generates a downloadable PDF report summarizing the recommendations.
- **Interactive Visualizations**: Displays a pie chart of the recommended asset allocation.
- **Regulatory Safeguards**: Ensures compliance with predefined financial rules and safeguards.

## Technologies Used
- **Python**: Core programming language.
- **Streamlit**: For building the interactive web application.
- **NumPy & Pandas**: For data manipulation and processing.
- **Scikit-learn**: For scaling and machine learning model predictions.
- **FPDF**: For generating PDF reports.
- **Pytest**: For testing the application.
- **Flake8**: For linting and code style enforcement.

## Project Structure

├── app.py # Main application file \
├── data_generator.py # Script to generate synthetic user data \
├── explain_and_validate.py # Functions for explanations and validations \
├── model_training.py # Script for training the asset allocation model \
├── validation.py # Data validation classes \
├── requirements.txt # Python dependencies \
├── Dockerfile # Docker configuration \
├── docker-compose.yml # Docker Compose configuration \
├── logs/ # Logs directory \
│ └── recommendation_logs.csv
├── reports/ # Directory for generated reports \
├── synthetic_asset_allocation_dataset.csv # Synthetic dataset \
├── asset_allocation_model.pkl # Pre-trained machine learning model \
├── scaler.pkl # Scaler for feature normalization \
└── test.py # Unit tests

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Tushar041100/NetWorth-Tracker.git
   cd NetWorth-Tracker

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

## Usage
1. Open the application in your browser.
2. Fill out the form with your financial details.
3. Submit the form to get personalized asset allocation recommendations.
4. View the recommendations, explanations, and pie chart.
5. Download the PDF report for future reference.      

## Testing
   pytest test.py

## CI/CD Pipeline
The project includes a GitHub Actions workflow for CI/CD. It runs tests, lints the code, and deploys the application.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
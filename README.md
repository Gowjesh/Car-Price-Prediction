# Car Price Prediction

## Project Overview

Car Price Prediction is a Machine Learning project that estimates the selling price of a used car based on various vehicle characteristics such as brand, model, manufacturing year, fuel type, transmission, kilometers driven, engine specifications, owner details, and other important features.

The objective of this project is to help buyers and sellers determine a fair market price for a vehicle by using historical car sales data and machine learning techniques.

---

## Features

- Data preprocessing and cleaning
- Missing value handling
- Duplicate record removal
- Outlier detection and removal
- Feature engineering
- Categorical feature encoding
- Feature scaling
- Training multiple regression models
- Model evaluation using regression metrics
- Streamlit web application for price prediction
- Model saving and loading using Joblib

---

## Dataset

The dataset contains information about used cars, including:

- Make
- Model
- Manufacturing Year
- Kilometer Driven
- Fuel Type
- Transmission
- Seller Type
- Owner
- Engine
- Drivetrain
- Color
- Location
- Selling Price (Target Variable)

Target Variable:

- Selling Price

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removed duplicate records
- Handled missing values
- Removed outliers using the IQR method
- Encoded categorical variables
- Standardized numerical features
- Selected relevant features
- Split the dataset into training and testing sets

---

## Feature Engineering

Feature engineering techniques were applied to improve model performance, including:

- Vehicle Age = Current Year - Manufacturing Year
- Encoded categorical variables
- Feature selection
- Scaling numerical features

---

## Machine Learning Models

The following regression models were trained and compared:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

The best-performing model was selected based on evaluation metrics.

---

## Evaluation Metrics

The models were evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Streamlit Application

The project includes a Streamlit application that allows users to:

- Enter car details
- Predict the estimated selling price
- Display the predicted value instantly

Run the application using:

```bash
streamlit run app.py
```

---

## Project Structure

```
Car-Price-Prediction/
│
├── dataset/
│   └── car_data.csv
│
├── models/
│   ├── car_price_model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Car-Price-Prediction.git
```

Navigate to the project directory:

```bash
cd Car-Price-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Workflow

1. Load the car price dataset.
2. Perform data preprocessing and cleaning.
3. Apply feature engineering.
4. Train multiple regression models.
5. Evaluate model performance.
6. Save the best-performing model.
7. Load the trained model into the Streamlit application.
8. Enter vehicle details to predict the estimated selling price.

---

## Future Enhancements

- Improve prediction accuracy using advanced ensemble models.
- Deploy the application on cloud platforms.
- Add interactive dashboards and visualizations.
- Integrate real-time used car market data.
- Support batch predictions using CSV file uploads.

## Output

<img width="1919" height="914" alt="image" src="https://github.com/user-attachments/assets/b34b01ab-97d4-4f37-af05-3b314af77dfc" />


<img width="1142" height="911" alt="image" src="https://github.com/user-attachments/assets/b3eaf7a7-e443-4a97-9e05-3b296c59b6b4" />

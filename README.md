# Smart-Room-Occupancy-Estimation
Predicting real-time room occupancy using environmental sensor data and machine learning in R. Achieved 96.2% accuracy with Gradient Boosting. Project includes EDA, feature engineering, SMOTE balancing, and model comparison.

## Project Overview

- **Goal**: Predict room occupancy using environmental sensors
- **Tech Stack**: R, SMOTE, caret, base plotting libraries
- **Dataset**: Multi-sensor dataset with 10,000+ records (CO₂, temp, light, sound, PIR motion)
- **Best Model**: Gradient Boosting Machine (GBM)

---

## Key Features

- Performed **EDA** to visualize trends in occupancy and sensor readings
- Engineered time-based features (hour, weekday, part of day, is_weekend)
- Addressed class imbalance with **SMOTE**
- Trained 5 models:
  - Poisson Regression (baseline)
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - **Gradient Boosting Machine (GBM)**
- Evaluated with Accuracy, F1-Score, Precision, Recall, and Balanced Accuracy

---

## Results

| Model             | Accuracy | F1 Score | Recall | Precision |
|-------------------|----------|----------|--------|-----------|
| Poisson Regression | 81.6%   | 0.926    | 52.3%  | 60.9%     |
| Decision Tree      | 83.9%   | 0.947    | 56.8%  | 61.2%     |
| Random Forest      | 90.0%   | 0.957    | 69.1%  | 77.2%     |
| **GBM**            | 96.2%   | 0.987    | 90.1%  | 87.2%     |
| SVM                | 89.6%   | 0.967    | 75.3%  | 69.7%     |

GBM showed the strongest performance in both majority and minority occupancy levels.

---

## Insights

- **CO₂ and Sound** were the most correlated features with occupancy.
- **Temperature** and **light intensity** had weaker standalone signals but added value when combined.
- Weekday patterns and time-of-day influenced occupancy trends.
- PIR sensors were effective for activity detection during working hours.

---

## Technologies Used

- R (caret, DMwR, ggplot2)
- SMOTE (for class balancing)
- Multiple classification models
- Sensor-based time series data
- Visualization (histograms, heatmaps, scatter plots, box plots)

---

## Files Included

- `Prsentation.pptx`: Prsentation including data description, EDA, visualizations, model evaluations, and insights
- `Code.R`: R script containing model training and evaluation code
- `Occupancy_Estimation.csv`: Dataset
- `README.md`: Project summary

---

## How to Run

1. Clone the repo  
2. Load the dataset and install required R packages  
3. Run `Code.R` for preprocessing, training, and evaluation  

---

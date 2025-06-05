# FIFA 23 Player Rating Prediction

A machine learning project that predicts FIFA 23 player overall ratings using regression analysis. Built with Python in Google Colab, achieving 89.2% accuracy with XGBoost.

## Project Overview

This project aims to predict FIFA 23 player overall ratings using various machine learning models. The complete process includes data analysis, preprocessing, model training, and evaluation.

The main goal is to build an accurate ML model that can predict a player's overall rating based on their various attributes and statistical data.

## Dataset Information

Source: Kaggle FIFA Player Stats Database
Size: 18,000+ players
Features: 100+ different attributes including technical skills, physical data, positional statistics, and personal information
Target: Overall Rating (0-100 scale)

The dataset includes personal data, technical skills like shooting and passing, physical attributes such as speed and strength, positional statistics, and market value information.

## Data Preprocessing

Data cleaning involved removing unnecessary columns like IDs and URLs, handling missing values, and correcting data types. Categorical data was converted to numerical format using LabelEncoder for positions, clubs, and nationalities.

Feature scaling was applied using StandardScaler to normalize all attributes. Outliers were detected and removed using the IQR method, removing only extreme values (approximately 2-3% of data).

## Exploratory Data Analysis

The analysis included distribution analysis of target and feature variables using histograms, correlation analysis with correlation matrices and heatmaps, outlier visualization with boxplots, and identification of important features using bar charts.

## Machine Learning Models

Five regression models were tested:

Linear Regression serves as a simple baseline model that's easy to interpret. Random Forest Regressor uses ensemble methods with 100 trees and resists overfitting. Gradient Boosting Regressor employs sequential boosting for high accuracy. Support Vector Regressor uses kernel-based methods for complex data. XGBoost Regressor represents advanced gradient boosting with state-of-the-art results.

## Model Evaluation

Models were evaluated using R² Score to show how much the model explains target variance, RMSE as root mean square error, MAE as mean absolute error, and 5-Fold Cross Validation to check model stability.

## Results

| Model | R² Score | RMSE | MAE | Cross-Val R² |
|-------|----------|------|-----|--------------|
| XGBoost | 0.892 | 2.34 | 1.87 | 0.885 |
| Random Forest | 0.879 | 2.48 | 1.92 | 0.874 |
| Gradient Boosting | 0.864 | 2.63 | 2.05 | 0.859 |
| SVR | 0.721 | 3.76 | 2.89 | 0.715 |
| Linear Regression | 0.687 | 3.98 | 3.12 | 0.682 |

XGBoost Regressor performed best with the highest R² score of 0.892, lowest RMSE of 2.34, and best cross-validation score of 0.885, showing stability across different data splits.

The top 10 most important features are Potential, Reactions, Composure, Ball Control, Short Passing, Long Passing, Dribbling, Vision, Finishing, and Shot Power.

## Technical Details

Regression was chosen because overall rating is a continuous value from 0-100. An 80%-20% train-test split was used with stratification to maintain balanced distribution. Feature engineering included scaling, encoding, and filling missing values with median or "Unknown" labels.

## Future Improvements

Better feature engineering could include new attributes like age groups and BMI. More advanced models such as neural networks and ensemble combinations could be explored. Hyperparameter tuning using GridSearchCV or RandomizedSearchCV would optimize performance. Domain knowledge from football experts could provide valuable insights.

## Required Technologies

Python libraries needed include pandas 1.3.0+, numpy 1.21.0+, scikit-learn 1.0.0+, matplotlib 3.4.0+, seaborn 0.11.0+, xgboost 1.5.0+, and kagglehub 0.1.0+.

Environment options include Google Colab (recommended), Jupyter Notebook, or Python 3.8+.

## Business Value

This project has applications in player scouting for discovering new talent, team management for optimal formation, transfer decisions for better price evaluation, and player development for determining correct development paths.

The return on investment includes more accurate player evaluation, reduced risk in transfer markets, and data-driven decision making.

## Conclusion

This project demonstrates that machine learning can provide highly accurate predictions for FIFA player overall ratings. XGBoost outperformed other models with 89.2% explained variance.

Key findings show that technical skills like passing and dribbling have the most impact on ratings, potential has the greatest significance, ensemble models work significantly better, and feature engineering greatly influences results.

The project can serve as a foundation for real football analytics and more complex sports prediction systems.

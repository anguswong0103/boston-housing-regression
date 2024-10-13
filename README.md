# Boston Housing Regression Project

## Introduction
This project demonstrates various machine learning models applied to the **Boston Housing dataset**. The goal is to predict the median value of homes (`MEDV`) based on various features such as the number of rooms, property tax rate, and more.

Several regression algorithms have been implemented and compared to evaluate their performance in predicting housing prices.

## Features
- **Data Preprocessing**: Scales the features using `MinMaxScaler` for normalization.
- **Machine Learning Models**:
  - Linear Regression
  - K-Nearest Neighbors (KNN) Regressor
  - Decision Tree Regressor
  - Random Forest Regressor
- **Model Evaluation**: 
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)

## Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/boston-housing-regression.git
   cd boston-housing-regression
   ```

2. Install the necessary packages:
   You can install the dependencies using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install Jupyter Notebook to run the `.ipynb` file:
   ```bash
   pip install notebook
   ```

4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook boston_housing_regression.ipynb
   ```

## Dataset
The **Boston Housing dataset** is publicly available and can be loaded directly using the `sklearn.datasets` module. It contains 506 instances, each representing a different Boston suburb. The target variable is the median value of homes (`MEDV`), and there are 13 features related to housing conditions.

- **Features**:
  - CRIM: Per capita crime rate by town
  - RM: Average number of rooms per dwelling
  - TAX: Full-value property-tax rate per $10,000
  - **...and others**

- **Target**: `MEDV` (Median value of owner-occupied homes in $1000s)

## Usage
### Running the Models
You can directly run the models and evaluate their performance using the notebook:

1. **Linear Regression**:
   ```python
   from sklearn.linear_model import LinearRegression
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

2. **Evaluating Models**:
   After training, evaluate the model using metrics such as RMSE and R-squared.
   ```python
   from sklearn.metrics import mean_squared_error, r2_score
   pred = model.predict(X_test)
   mse = mean_squared_error(y_test, pred)
   r2 = r2_score(y_test, pred)
   ```

### Visualizing Results
The notebook includes visualizations of the predictions versus actual values, as well as the prediction errors.

## Model Performance
The performance of different models is compared using the test set, focusing on **RMSE** and **R-squared**. Here is a sample output:

| Model                  | RMSE   | R-squared |
|-------------------------|--------|-----------|
| Linear Regression        | 4.84   | 0.76      |
| K-Nearest Neighbors      | 4.64   | 0.78      |
| Decision Tree Regressor  | 4.18   | 0.82      |
| Random Forest Regressor  | 2.88   | 0.92    |

## Future Improvements
- **Hyperparameter Tuning**: Apply techniques such as GridSearchCV or RandomizedSearchCV to improve model performance.
- **Cross-Validation**: Implement k-fold cross-validation to ensure model robustness.
- **Feature Engineering**: Perform more sophisticated feature engineering to extract new insights from the dataset.

---
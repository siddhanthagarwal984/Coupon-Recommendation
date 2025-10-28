# Coupon Recommendation System

## Overview
This project implements a machine learning-based coupon recommendation system using various classification algorithms. The goal is to predict whether a user will accept a coupon based on demographic, behavioral, and contextual features. The project utilizes a dataset containing user attributes and coupon details to train and evaluate multiple models, ultimately selecting the best-performing one for recommendations.

The analysis is conducted in a Jupyter Notebook (`Siddhanth_Coupon_Recommendation.ipynb`), which includes data exploration, preprocessing, feature engineering, model training, evaluation, and hyperparameter tuning.

## Dataset
The dataset used is `DS_DATA.csv`, which contains information about users and coupons. Key features include:
- **Demographic features**: Age, gender, marital status, education, occupation, income.
- **Behavioral features**: Frequency of visits to bars, coffee houses, carry-away restaurants, and restaurants in different price ranges.
- **Contextual features**: Destination, passenger type, weather, coupon type, expiration time, distance to coupon location.
- **Target variable**: `Accept(Y/N?)` - Binary indicator of whether the coupon was accepted (1 for yes, 0 for no).

The dataset has 12,684 rows and multiple columns. It includes missing values and categorical variables that require preprocessing.

## Dependencies
The following Python libraries are required to run the notebook:
- `os`
- `sys`
- `sklearn` (including submodules like `preprocessing`, `model_selection`, `impute`, `linear_model`, `tree`, `ensemble`, `neighbors`, `svm`, `naive_bayes`, `metrics`)
- `matplotlib`
- `xgboost`
- `numpy`
- `pandas`
- `seaborn`
- `warnings`

Install the dependencies using pip:
```
pip install scikit-learn matplotlib xgboost numpy pandas seaborn
```

## Installation and Setup
1. Clone or download the repository to your local machine.
2. Ensure Python 3.x is installed.
3. Install the required dependencies as listed above.
4. Place the `DS_DATA.csv` file in the same directory as the notebook.
5. Open the notebook in Jupyter Lab or Jupyter Notebook.

## Project Structure
- `Siddhanth_Coupon_Recommendation.ipynb`: Main notebook containing the entire analysis.
- `DS_DATA.csv`: Dataset file (not included in the repository; must be provided separately).
- `README.md`: This file.
- `Siddhanth_Coupon_Recommendation_PPT (1).pptx`: Presentation slides (optional).

## Data Exploration and Preprocessing
### Initial Data Loading and Inspection
- Load the dataset using `pandas.read_csv('DS_DATA.csv')`.
- Display basic information: `df.info()`, `df.describe()`, `df.shape`, `df.columns`.
- Check target variable distribution: `df['Accept(Y/N?)'].value_counts(normalize=False)`.

### Handling Duplicates
- Identify duplicated rows: `df.duplicated().sum()`.
- Since duplicates are minimal (count of 2), they are retained as they might represent valid data points.

### Missing Values
- Check for null values: `df.isnull().sum()`.
- The 'car' column has 99% missing data, so it is dropped: `df.drop(columns='car', axis=1)`.
- Separate categorical and numerical columns.
- Impute numerical columns with mean using `SimpleImputer(strategy='mean')`.
- Impute categorical columns with most frequent value using `SimpleImputer(strategy='most_frequent')`.
- Concatenate the imputed dataframes back together.

### Correlation Analysis
- Compute correlation matrix for numerical features: `num.corr()`.
- Visualize with a heatmap using `seaborn.heatmap()`.

#### Correlation Heatmap
The correlation heatmap is a visual representation of the pairwise correlations between numerical features in the dataset. It uses a color scale (typically from blue to red) where:
- Dark blue indicates strong negative correlation (close to -1).
- White indicates no correlation (close to 0).
- Dark red indicates strong positive correlation (close to 1).

Key observations from the heatmap:
- Features like `toCoupon_GEQ15min`, `toCoupon_GEQ25min`, and `direction_same` show correlations with distance-related variables.
- Behavioral features (e.g., frequency of visits to restaurants) may show moderate correlations with demographic features like income or age.
- The target variable `Accept(Y/N?)` is not included in this heatmap as it is categorical, but correlations with numerical features can inform feature selection.

This plot helps identify multicollinearity, which could affect model performance, and guides feature engineering decisions.

### Feature Engineering
- **One-Hot Encoding**: Apply to nominal categorical variables: `destination`, `passanger`, `weather`, `coupon`, `gender`, `maritalStatus`, `education`, `occupation`. Use `pd.get_dummies()` with `drop_first=True` and `dtype=int`.
- **Ordinal Encoding**: Manually map ordinal variables to numerical scales:
  - `expiration`: '2h' -> 2, '1d' -> 24
  - `age`: 'below21' -> 0, '21' -> 1, '26' -> 2, '31' -> 3, '36' -> 4, '41' -> 5, '46' -> 6, '50plus' -> 7
  - Frequency variables (`Bar`, `CoffeeHouse`, `CarryAway`, `RestaurantLessThan20`, `Restaurant20To50`): 'never' -> 0, 'less1' -> 1, '1~3' -> 2, '4~8' -> 3, 'gt8' -> 4
- **Income Mapping**: Convert income ranges to midpoints:
  - 'Less than $12500' -> 6250
  - '$12500 - $24999' -> 18750
  - And so on, up to '$100000 or More' -> 112500
- Drop constant column: `toCoupon_GEQ5min` (all values are the same).

### Final Dataset Preparation
- Split into features (`x`) and target (`y`).
- Train-test split: `train_test_split(x, y, test_size=0.2, random_state=22, stratify=y)` to maintain class balance.
- Clean column names for model compatibility: Replace special characters with underscores and convert to strings.

## Modeling
Multiple classification models are trained and evaluated:
1. **Logistic Regression**: `LogisticRegression()`
2. **Decision Tree**: `DecisionTreeClassifier()`
3. **Random Forest**: `RandomForestClassifier()`
4. **XGBoost**: `XGBClassifier(colsample_bytree=0.7, learning_rate=0.1, max_depth=9, n_estimators=300, reg_lambda=10, subsample=0.8)`
5. **K-Nearest Neighbors**: `KNeighborsClassifier()`
6. **Support Vector Machine**: `SVC()`
7. **Naive Bayes**: `GaussianNB()`
8. **Voting Classifier**: Ensemble of all above models using `VotingClassifier()`.

### Evaluation Metrics
- Accuracy score for each model.
- Confusion matrix and classification report for the best model.
- Cross-validation scores for XGBoost.

### Model Comparison
- Create a dataframe with model names and accuracies.
- Visualize with bar plots using `seaborn.barplot()`.
- Sort and display the best model.

#### Model Accuracy Comparison Bar Plot
This bar plot compares the accuracy scores of all trained models. The x-axis lists the model names (LogitRegression, Dtree, RForest, KNN, SVM, NBT, Voting, XGBoost), and the y-axis shows the accuracy percentage (ranging from 0 to 1).

Key observations:
- XGBoost typically achieves the highest accuracy (e.g., around 0.75), followed by Random Forest and Voting Classifier.
- Simpler models like Logistic Regression and Naive Bayes have lower accuracies (e.g., around 0.65-0.70).
- The plot is sorted in descending order of accuracy to highlight the best performers.
- This visualization helps in quickly identifying the top models for further tuning.

The plot is generated twice: once unsorted and once sorted, with x-axis labels rotated 90 degrees for readability.

### Hyperparameter Tuning
- Use `GridSearchCV` on XGBoost with parameters:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 6, 9]
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `subsample`: [0.7, 0.8, 1]
  - `colsample_bytree`: [0.7, 0.8, 1]
  - `reg_lambda`: [1, 5, 10]
- Best parameters and cross-validation accuracy are printed.
- Evaluate tuned model on test set with classification report and confusion matrix.

#### Confusion Matrix
The confusion matrix is a table that summarizes the performance of the tuned XGBoost model on the test set. It shows the counts of:
- True Positives (TP): Correctly predicted accepts.
- True Negatives (TN): Correctly predicted non-accepts.
- False Positives (FP): Incorrectly predicted accepts (actually no).
- False Negatives (FN): Incorrectly predicted non-accepts (actually yes).

The matrix is visualized using `ConfusionMatrixDisplay` from sklearn, with annotations showing the exact counts. This helps assess the model's ability to distinguish between the two classes, especially important given the class imbalance.

### Cross-Validation
- Perform 10-fold cross-validation on XGBoost: `cross_val_score(xgb_model, x_train, y_train, cv=10, scoring='accuracy')`.
- Plot cross-validation scores.

#### Cross-Validation Scores Plot
This line plot displays the accuracy scores from each of the 10 cross-validation folds on the training set. The x-axis represents the fold number (1 to 10), and the y-axis shows the accuracy score for that fold.

Key observations:
- The scores vary slightly across folds, indicating model stability.
- The mean accuracy is calculated and printed, providing an estimate of the model's generalization performance.
- If the scores are consistent, it suggests the model is not overfitting; high variance might indicate the need for more data or regularization.

## Results
- **Class Distribution**: The target variable is imbalanced (approximately 57% no, 43% yes).
- **Model Accuracies** (example from run):
  - XGBoost: Highest accuracy (e.g., ~0.75)
  - Random Forest: ~0.73
  - Voting Classifier: ~0.72
  - Others vary.
- **Best Model**: XGBoost after tuning.
- **Classification Report**: Precision, recall, F1-score for both classes.
- **Confusion Matrix**: Visualized to show true positives, false positives, etc.

## How to Run
1. Ensure all dependencies are installed.
2. Open `Siddhanth_Coupon_Recommendation.ipynb` in Jupyter.
3. Run cells sequentially from top to bottom.
4. The notebook will load data, preprocess it, train models, and display results.
5. Outputs include plots, tables, and metrics.

## Key Insights
- XGBoost performs best due to its ability to handle complex interactions.
- Feature engineering significantly improves model performance.
- Class imbalance should be addressed in future iterations (e.g., using SMOTE).
- Contextual features like weather and destination are important predictors.

## Future Improvements
- Address class imbalance with techniques like oversampling or undersampling.
- Feature selection to reduce dimensionality.
- Deploy the model as a web service for real-time recommendations.
- Experiment with deep learning models.

## Author
Siddhanth

## License
This project is for educational purposes. Please cite if used.

# Model Card - LogisticalRegression on Census Data
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Used a Logistical Regression model
- Trained on 1994 Census Data (https://archive.ics.uci.edu/dataset/20/census+income)
- Goal is to predict wheteher a person makes over 50k a year based on other variables (age, workclass, education, etc)

## Intended Use
- Used in an Educational Setting for Udacity
- Purpose is to explore machine learning deployment in pipelines
- Could also be used for additional analysis or insights in income levels, and repeatable with newer census data

## Training Data
- 1994 Census Data
- 32,561 rows
- 14 attributes
- Categorical and numerical data

## Evaluation Data
- Test set split from the original csv.
- Composed of 20% of the original csv data

## Metrics
- Precision, recall, and F1 score. 
- On the test data, the model achieved a precision of 0.7110, a recall of 0.2674, and an F1 score of 0.3886.

## Ethical Considerations
- No ethical considerations on the model itself
- Census data that it was trained on may have included biases on collection. Predictions on this model should be used mainly for educational purposes of ML modeling and not used to make decisions for policy on.

## Caveats and Recommendations
- When it comes to the data, a lot of factors outside the collection may have contributed to income levels.
- Other models should be tested alongside LogisticalRegression to improve accuracy and performance
- Hyperparameters could be added to the model 

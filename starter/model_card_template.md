# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Samuel Amico created two models. The first is the RandomForest model
that uses as paramters = max_depth=15, min_samples_split=4, min_samples_leaf=3,random_state=0
The second model is the Gradient Boosting Classifier, with n_estimators: (5, 10),
learning_rate = (0.1, 0.01), max_depth= [2, 3], max_features: ("auto", "log2").

Both models are from scikit-learn latest version

## Intended Use
This model should be used to predict the salary based on some personal features

## Training Data
Selected 80% of data for training phase.

## Evaluation Data
Selected 20% of data for evaluation phase.

## Metrics
The GB model was evaluated using fbeta : 0.5729335494327391 precision : 0.7855555555555556 recall : 0.45089285714285715
and the RF model was evaluated using MAE = 0.16183 R2 = 0.4653012463

## Ethical Considerations
Used publicly available Census Bureau data https://archive.ics.uci.edu/ml/datasets/census+income. The dataset contains data that could potentially discriminate against people, sensity information.

## Caveats and Recommendations
Improve the model and change more the parametes, also use a big dataset.
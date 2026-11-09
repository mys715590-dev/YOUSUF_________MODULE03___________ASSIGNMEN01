ANN Salary Regression
Project overview
This classroom project changes the usual churn-classification workflow into a regression task. The model predicts EstimatedSalary as a continuous number from the customer information in Churn_Modelling.csv.
This is a learning exercise only. The dataset and model are not suitable for real banking, credit, employment, or salary decisions.
Dataset
The project uses the 10,000-row Churn Modelling dataset. EstimatedSalary is the target. RowNumber, CustomerId, and Surname are removed because they are identifiers or names rather than useful customer measurements. The remaining columns, including Exited, are used as input features.
Preprocessing
Split data into training and test sets (80/20).
Identify numeric and categorical input columns.
Impute missing numeric values with the training median and standardize them.
Impute missing categorical values with the most frequent training value and one-hot encode them.
Fit the preprocessing pipeline on training data only; transform the test data with that fitted pipeline.
Save the fitted pipeline and reuse it in the Streamlit app.
ANN architecture
The network has dense hidden layers with 64, 32, and 16 units using ReLU activation, with dropout after the first hidden layer. The output layer has one neuron and a linear activation because salary is a continuous value. The model is trained with mean squared error (MSE) loss and Adam optimizer. Early stopping monitors validation loss.
Regression evaluation
The training script reports:
MAE: average absolute prediction error, expressed in the same currency units as EstimatedSalary.
RMSE: square root of mean squared error. Larger errors receive a stronger penalty, so RMSE can be greater than MAE.
R²: compares model predictions with a baseline that predicts the test-set mean. It can be negative when the model performs worse than that baseline.
Run the training script to generate the actual held-out test metrics. Do not insert guessed metric values into the report.
Run locally
Put Churn_Modelling.csv in the project root beside train_model.py and app.py.
Create and activate a virtual environment (recommended).
Install packages:
pip install -r requirements.txt
Train the model and create artifacts:
python train_model.py
Start the app:
streamlit run app.py
Deploy on Streamlit Community Cloud
Push this repository to GitHub.
Ensure app.py, requirements.txt, Churn_Modelling.csv, and the generated artifacts/ files are committed.
In Streamlit Community Cloud, choose the repository and set the main file path to app.py.
Deploy and test the input form and prediction.
Add the public app URL and GitHub repository URL to your submission.
Do not deploy before running train_model.py; the app needs the saved model and preprocessing files.
Classification vs regression
A classification model predicts a class or category, such as whether a customer will leave. This project instead predicts a continuous numeric target, EstimatedSalary. The ANN therefore uses one linear output neuron and MSE loss instead of a sigmoid classification output and binary cross-entropy. Evaluation uses MAE, RMSE, and R² rather than accuracy, precision, recall, or a confusion matrix.
Limitations
The dataset is synthetic/educational and may not represent real customers or salary patterns. A model can learn dataset-specific correlations that do not generalize. Predictions may be affected by sampling, feature choices, data quality, and random variation. The output should not be interpreted as a real person's actual salary or used to make banking decisions.
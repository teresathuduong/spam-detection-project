# spam-detection-project

This project builds a spam classifier using machine learning techniques to distinguish between spam and ham (non-spam) messages. It includes data preprocessing, feature extraction, model training, evaluation, and visualization.

## 📂 Dataset
- Source: Kaggle
- Columns Used:
- v1: Label (ham or spam)
- v2: Text message

## 🧹 Preprocessing
- Lowercasing text
- Removing digits and punctuation
- Removing basic stopwords
- Token cleanup and normalization

## 🧠 Models Used
- Logistic Regression (GridSearchCV)
- Naive Bayes (GridSearchCV)
- Random Forest (RandomizedSearchCV)
Each model is tuned using cross-validation and evaluated on accuracy, precision, recall, F1 score, and ROC AUC.

## 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix
ROC curves are plotted to compare model performance visually.

![ROC Curve](document/roc_curve)

## 🛠️ Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Regex, String manipulation

## 🚀 How to Run
- Clone the repo:
git clone https://github.com/teresathuduong/spam-detection.git
cd spam-detection
- Install dependencies:
pip install -r requirements.txt
- Run the script:
python spam_detection.py


## 📌 Notes
- TF-IDF vectorization is used with bigrams and a max feature limit of 3000.
- The project saves the ROC curve as roc_curve.png.

# train_naive_bayes.py
import os
import json
import argparse
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def main(args):
    os.makedirs('models', exist_ok=True)

    # Load training data
    df = pd.read_csv(args.data_path)
    print("Columns:", df.columns.tolist())

    # Use 'input' as the ONLY feature
    feature_cols = ['input']

    # Convert into list of dicts
    X_dict = df[feature_cols].astype(str).to_dict(orient='records')


    # Convert categorical features to one-hot vectors
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(X_dict)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df['target'].astype(str))

    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X, y)

    # Save model and encoders
    joblib.dump(clf, 'models/naive_bayes.joblib')
    joblib.dump(vec, 'models/vectorizer.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')

    # Save metadata
    meta = {'feature_cols': feature_cols}
    with open('models/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    print("Naive Bayes model trained and saved in 'models/' directory")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/train.csv', help='CSV file with training data')
    args = parser.parse_args()
    main(args)


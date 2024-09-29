# Import necessary libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load data
def load_data(file_path):
    positive_path = os.path.join(file_path, './rt-polarity.pos')
    negative_path = os.path.join(file_path, './rt-polarity.neg')
    
    with open(positive_path, 'r', encoding='ISO-8859-1') as pos_file:
        positive_sentences = pos_file.readlines()
    
    with open(negative_path, 'r', encoding='ISO-8859-1') as neg_file:
        negative_sentences = neg_file.readlines()
    
    # Create DataFrame with text and labels
    positive_labels = [1] * len(positive_sentences)
    negative_labels = [0] * len(negative_sentences)
    
    data = pd.DataFrame({
        'text': positive_sentences + negative_sentences,
        'label': positive_labels + negative_labels
    })
    
    return data

# Preprocess data
def preprocess_data(data):
    data['text'] = data['text'].str.lower()  # Lowercase text
    data['text'] = data['text'].str.replace(r'[^\w\s]', '', regex=True)  # Remove punctuation
    return data

# Split data into training, validation, and test sets
def split_data(data):
    # Create train, validation, and test sets as per the requirements
    train_data = pd.concat([data[:4000], data[5331:9331]])
    val_data = pd.concat([data[4000:4500], data[9331:9831]])
    test_data = pd.concat([data[4500:], data[9831:]])
    
    return train_data, val_data, test_data

# Vectorize text using TF-IDF
def vectorize_text(train_data, val_data, test_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_val = vectorizer.transform(val_data['text'])
    X_test = vectorizer.transform(test_data['text'])
    
    return X_train, X_val, X_test, vectorizer

# Train the model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    
    precision = cr['1']['precision']
    recall = cr['1']['recall']
    f1_score = cr['1']['f1-score']
    
    return tp, tn, fp, fn, precision, recall, f1_score

# Main function
if __name__ == "__main__":
    # Load and preprocess data
    file_path = './'
    data = load_data(file_path)
    data = preprocess_data(data)

    # Split data
    train_data, val_data, test_data = split_data(data)

    # Vectorize text
    X_train, X_val, X_test, vectorizer = vectorize_text(train_data, val_data, test_data)

    # Train the model
    model = train_model(X_train, train_data['label'])

    # Evaluate the model
    tp, tn, fp, fn, precision, recall, f1_score = evaluate_model(model, X_test, test_data['label'])

    # Print evaluation metrics
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

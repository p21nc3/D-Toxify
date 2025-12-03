"""
YanClassifier.py: Classifies tweets as hate speech or non-hate speech.

Usage:
    YanClassifier.py [options]

Options:
    --test-path=<file>                          testing file
    --classifier-name=<str>                     name of classifier
    --model-path=<file>                         model directory path
    --export-results-path=<file>                testing report file
    --prediction-mode=<str>                     prediction mode (binary or probability)
    --rephrase=<bool>                           enable rephrasing for detected hate speech [default: False]
    --rephrase-model=<str>                      Hugging Face model name for rephrasing [default: gpt2]
    --rephrase-threshold=<float>                probability threshold for rephrasing in probability mode [default: 0.5]
"""

#===========================#
#        Imports            #
#===========================#

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import numpy as np
import sys
from docopt import docopt
import os
import pickle
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Import rephraser
# Add parent directories to path to find rephraser module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rephraser import create_rephraser
    REPHRASER_AVAILABLE = True
except ImportError as e:
    REPHRASER_AVAILABLE = False
    print(f"Warning: Rephraser module not available ({e}). Rephrasing will be disabled.")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

#===========================#
#        Variables          #
#===========================#

PREDICTION_MODE_BINARY = 'binary'
PREDICTION_MODE_PROBABILITY = 'probability'

#===========================#
#        Functions          #
#===========================#

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    for token in word_tokens:    
        if token.lower() not in stop_words and len(token) > 2:  # drops words with less than 3 characters
            result.append(lemmatize(token))
    return result

def preprocess_tweet(tweet):
    """Preprocesses a tweet for classification"""
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    processed = tokenize(result)
    # Convert list of tokens back to string for vectorizer
    processed_str = ' '.join(processed) if isinstance(processed, list) else processed
    return processed_str

def load_model(model_path):
    """Loads the classifier and vectorizer models"""
    clf_path = os.path.join(model_path, 'clf.pickle')
    vec_path = os.path.join(model_path, 'vec.pickle')
    
    if not os.path.exists(clf_path):
        raise FileNotFoundError(f"Model file not found: {clf_path}")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vec_path}")
    
    model = pickle.load(open(clf_path, "rb"))
    vec = pickle.load(open(vec_path, "rb"))
    
    return model, vec

def classify_texts(test_path, model_path, prediction_mode, 
                   enable_rephrase=False, rephrase_model=None, rephrase_threshold=0.5):
    """
    Classifies texts from a CSV file and optionally rephrases detected hate speech.
    
    Args:
        test_path: Path to CSV file with 'id' and 'text' columns
        model_path: Path to directory containing model files
        prediction_mode: 'binary' or 'probability'
        enable_rephrase: Whether to rephrase detected hate speech
        rephrase_model: Hugging Face model name for rephrasing
        rephrase_threshold: Probability threshold for rephrasing in probability mode
    
    Returns:
        DataFrame with id, text, label, and optionally rephrased_text columns
    """
    # Load the test data
    df = pd.read_csv(test_path)
    
    # Validate required columns
    if 'id' not in df.columns or 'text' not in df.columns:
        raise ValueError("CSV must contain 'id' and 'text' columns")
    
    # Load models
    model, vec = load_model(model_path)
    
    # Preprocess texts
    processed_texts = df['text'].apply(preprocess_tweet).tolist()
    
    # Vectorize
    vectorized = vec.transform(processed_texts)
    
    # Predict
    if prediction_mode == PREDICTION_MODE_BINARY:
        predictions = model.predict(vectorized)
        df['label'] = predictions.astype(int)
    elif prediction_mode == PREDICTION_MODE_PROBABILITY:
        probabilities = model.predict_proba(vectorized)[:, 1]
        df['label'] = probabilities
    else:
        raise ValueError(f"Unknown prediction mode: {prediction_mode}")
    
    # Rephrase detected hate speech if enabled
    if enable_rephrase and REPHRASER_AVAILABLE:
        print("Rephrasing detected hate speech...")
        rephraser = create_rephraser(model_name=rephrase_model)
        
        # Determine which texts to rephrase
        if prediction_mode == PREDICTION_MODE_BINARY:
            texts_to_rephrase = df['label'] == 1
        else:
            texts_to_rephrase = df['label'] >= rephrase_threshold
        
        # Initialize rephrased_text column
        df['rephrased_text'] = df['text'].copy()
        
        # Rephrase hate speech
        hate_texts = df.loc[texts_to_rephrase, 'text'].tolist()
        if hate_texts:
            print(f"Rephrasing {len(hate_texts)} hate speech instances...")
            rephrased = rephraser.rephrase_batch(hate_texts)
            df.loc[texts_to_rephrase, 'rephrased_text'] = rephrased
            print("Rephrasing complete!")
    
    # Select columns to return
    columns = ['id', 'text', 'label']
    if enable_rephrase and REPHRASER_AVAILABLE:
        columns.append('rephrased_text')
    
    return df[columns]

#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":
    args = docopt(__doc__)
    
    test_path = args['--test-path']
    classifier_name = args['--classifier-name']
    model_path = args['--model-path']
    export_results_path = args['--export-results-path']
    prediction_mode = args['--prediction-mode']
    
    # Parse rephrasing options
    enable_rephrase = args.get('--rephrase', 'False').lower() in ('true', '1', 'yes', 'on')
    rephrase_model = args.get('--rephrase-model') or None
    rephrase_threshold = float(args.get('--rephrase-threshold', '0.5'))
    
    print(f'Processing file: {test_path}')
    print(f'Using model from: {model_path}')
    print(f'Prediction mode: {prediction_mode}')
    print(f'Rephrasing enabled: {enable_rephrase}')
    if enable_rephrase and rephrase_model:
        print(f'Rephrasing model: {rephrase_model}')
    
    try:
        # Classify texts
        results_df = classify_texts(
            test_path, 
            model_path, 
            prediction_mode,
            enable_rephrase=enable_rephrase,
            rephrase_model=rephrase_model,
            rephrase_threshold=rephrase_threshold
        )
        
        # Export results
        results_df.to_csv(export_results_path, index=False)
        
        print(f'Results exported to: {export_results_path}')
        print(f'Total texts classified: {len(results_df)}')
        
        if prediction_mode == PREDICTION_MODE_BINARY:
            hate_count = results_df['label'].sum()
            print(f'Hate speech detected: {hate_count} ({hate_count/len(results_df)*100:.1f}%)')
        else:
            avg_prob = results_df['label'].mean()
            hate_count = (results_df['label'] >= rephrase_threshold).sum()
            print(f'Average hate probability: {avg_prob:.3f}')
            print(f'Texts above threshold ({rephrase_threshold}): {hate_count}')
            
    except Exception as e:
        print(f'Error during classification: {str(e)}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


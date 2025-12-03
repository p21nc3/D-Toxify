#=============================================================================#
#
#   Docker metatool for Automated Hate Tweet Detection
#   This script processes CSV files containing tweets and classifies them
#   as hate speech or non-hate speech.
#
#   Based on the structure of online-harms-docker metatool
#
#=============================================================================#

#===========================#
#        Imports            #
#===========================#

import datetime
import nltk
import numpy as np
import os
import pandas as pd
import subprocess
import sys
import time

# Download NLTK data
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

CLASSIFIERS = ['yan']

PREDICTION_MODE_BINARY = 'binary'
PREDICTION_MODE_PROBABILITY = 'probability'

ALL_RESULTS_PATH = 'results/'
pd.set_option('display.max_rows', 500)

#===========================#
#        Functions          #
#===========================#

def generate_reports(file, prediction_mode, enable_rephrase=False, 
                     rephrase_model=None, rephrase_threshold=0.5):
    """
    Generates a report for each trained model.
    
    Args:
        file: CSV file to process
        prediction_mode: 'binary' or 'probability'
        enable_rephrase: Whether to enable rephrasing
        rephrase_model: Hugging Face model name for rephrasing
        rephrase_threshold: Probability threshold for rephrasing
    """

    print('-----> Generating reports based on the trained models...')

    file_name = str(file).split('.')[0]  # Get file name
    
    for classifier in CLASSIFIERS:
        
        # Get classifier path
        classifier_path = 'classifiers/' + \
            str(classifier) + '/' + \
            str(classifier) + \
            str('_classifier.py')
        
        print('-------> Generating report for classifier `' + \
            str(classifier) + '`')
        
        # Get model path
        model_path = 'classifiers/' + classifier + '/models/'

        # Get results path            
        results_path = 'results/' + \
            classifier + '_' + \
            file_name + '_' + \
            'report.csv'

        # Build command arguments
        cmd_args = [
            # Set the path of the classifier
            classifier_path,                     
            # Set the name of the classifier           
            '--classifier-name', str(classifier),
            # Path to the model directory
            '--model-path', model_path,
            # Set the process (test)
            '--test-path', str('dataset/') + file,
            # Set the path of the export model
            '--export-results-path', results_path,
            # Sets the prediction mode of the classifier
            '--prediction-mode', prediction_mode,
        ]
        
        # Add rephrasing options if enabled
        if enable_rephrase:
            cmd_args.extend(['--rephrase', 'True'])
            if rephrase_model:
                cmd_args.extend(['--rephrase-model', rephrase_model])
            cmd_args.extend(['--rephrase-threshold', str(rephrase_threshold)])

        # Run the classifier against the data
        # Use 'python' in Docker, but sys.executable for local testing
        python_cmd = os.environ.get('PYTHON_CMD', sys.executable)
        subprocess.call([python_cmd] + cmd_args)


def generate_voting_ensemble(file, prediction_mode):
    """
    Generates an ensemble report based on all the results.
    For now, since we only have one classifier, this just copies the results.
    """

    print('-----> Generating ensemble report...')

    file_name = str(file).split('.')[0]  # Get file name
    file_path = ALL_RESULTS_PATH + 'ensemble_report' + '_' + file_name + '.csv'

    # Delete a previous ensemble file if it already exists 
    if os.path.isfile(file_path):
        os.remove(file_path)

    # Get all files in the directory (exclude ensemble reports)
    results = [f for f in os.listdir(ALL_RESULTS_PATH) 
               if not f.startswith('.') and f.endswith('.csv') 
               and not f.startswith('ensemble_report')]

    if not results:
        print('-------> No results found to create ensemble')
        return

    # For now, just use the first (and only) result
    # In the future, if multiple classifiers are added, this can be expanded
    df = pd.read_csv(ALL_RESULTS_PATH + results[0], index_col=None, header=0)
    
    # Rename the label column to match ensemble format
    if 'label' not in df.columns and 'prediction' in df.columns:
        df['label'] = df['prediction']
    
    # Select columns for ensemble report
    columns = ['id', 'text', 'label']
    if 'rephrased_text' in df.columns:
        columns.append('rephrased_text')
    
    ensemble_df = df[columns].copy()
    ensemble_df.to_csv(file_path, index=False)
    
    print('-------> Ensemble report created: ' + file_path)


#===========================#
#           Main            #
#===========================#

if __name__ == "__main__":

    # Parse command-line arguments
    prediction_mode = PREDICTION_MODE_BINARY
    enable_rephrase = False
    rephrase_model = None
    rephrase_threshold = 0.5
    
    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-p' or arg == '--probability':
        prediction_mode = PREDICTION_MODE_PROBABILITY
        elif arg == '-r' or arg == '--rephrase':
            enable_rephrase = True
        elif arg == '--rephrase-model' and i + 1 < len(sys.argv):
            rephrase_model = sys.argv[i + 1]
            i += 1
        elif arg == '--rephrase-threshold' and i + 1 < len(sys.argv):
            rephrase_threshold = float(sys.argv[i + 1])
            i += 1
        elif arg == '--help' or arg == '-h':
            print("""
Hate Speech Detection and Rephrasing Tool

Usage:
    docker.py [options]

Options:
    -p, --probability              Use probability mode instead of binary mode
    -r, --rephrase                 Enable rephrasing for detected hate speech
    --rephrase-model MODEL         Hugging Face model name for rephrasing (default: gpt2)
    --rephrase-threshold FLOAT     Probability threshold for rephrasing (default: 0.5)
    -h, --help                     Show this help message

Examples:
    # Binary mode without rephrasing
    python docker.py
    
    # Probability mode with rephrasing
    python docker.py -p -r
    
    # Probability mode with custom rephrasing model and threshold
    python docker.py -p -r --rephrase-model distilgpt2 --rephrase-threshold 0.7
            """)
            sys.exit(0)
        i += 1

    # Start timer
    start = datetime.datetime.now()

    # Print configuration
    print('\n' + '='*60)
    print('Hate Speech Detection Configuration')
    print('='*60)
    print(f'Prediction mode: {prediction_mode}')
    print(f'Rephrasing enabled: {enable_rephrase}')
    if enable_rephrase:
        print(f'Rephrasing model: {rephrase_model or "gpt2 (default)"}')
        print(f'Rephrasing threshold: {rephrase_threshold}')
    print('='*60 + '\n')

    # Reading
    print('Scanning for a .csv file in folder `dataset`...')

    # Create results directory if it doesn't exist
    os.makedirs(ALL_RESULTS_PATH, exist_ok=True)

    files = [f for f in os.listdir('dataset') 
             if not f.startswith('.') and f.endswith('.csv')]

    if not files:
        print('No CSV files found in dataset directory!')
        print('Please place a CSV file with "id" and "text" columns in the dataset/ directory.')
        sys.exit(1)

    for file in files:
        print('===> File to be labelled: ' + str(file))
        generate_reports(file, prediction_mode, enable_rephrase, 
                        rephrase_model, rephrase_threshold)
        generate_voting_ensemble(file, prediction_mode)

    print('Reporting complete: Results can be found in folder `results`')

    # End timer
    end = datetime.datetime.now()

    # Print results
    print("\nTotal time: " + str(end - start))

#===========================#
#       End of Script       #
#===========================#


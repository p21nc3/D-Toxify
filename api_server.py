from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import pickle
import re
import nltk
from typing import Any
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import rephraser
try:
    from rephraser import create_rephraser
    REPHRASER_AVAILABLE = True
except ImportError:
    REPHRASER_AVAILABLE = False
    print("Warning: Rephraser module not available. Rephrasing will be disabled.")

# Import worker queue
try:
    from worker_queue import WorkerQueue
    WORKER_QUEUE_AVAILABLE = True
    print("✓ Worker queue module imported successfully")
except ImportError as e:
    WORKER_QUEUE_AVAILABLE = False
    print(f"Warning: Worker queue module not available: {e}")
    print("Queue endpoints will be disabled.")
except Exception as e:
    WORKER_QUEUE_AVAILABLE = False
    print(f"Warning: Error importing worker queue module: {e}")
    import traceback
    traceback.print_exc()
    print("Queue endpoints will be disabled.")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Enable CORS for browser extension

@app.after_request
def add_cors_headers(response):
    """Add CORS and Private Network Access headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    return response

# Download NLTK data if needed
def ensure_nltk_data():
    """Ensure all required NLTK data is downloaded"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading stopwords corpus...")
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("Downloading wordnet corpus...")
        nltk.download('wordnet', quiet=True)
    
    # Force WordNet to initialize properly
    try:
        from nltk.corpus import wordnet as wn
        # Try to access WordNet to ensure it's loaded
        _ = len(wn.synsets('test'))
        print("WordNet initialized successfully")
    except Exception as e:
        print(f"Warning: WordNet initialization issue: {e}")
        print("Attempting to re-download WordNet...")
        try:
            nltk.download('wordnet', quiet=True, force=True)
            from nltk.corpus import wordnet as wn
            _ = len(wn.synsets('test'))
            print("WordNet re-initialized successfully")
        except Exception as e2:
            print(f"Error re-initializing WordNet: {e2}")

# Initialize NLTK data at startup
ensure_nltk_data()

# Global variables for model
classifier_model = None
vectorizer = None
rephraser = None
worker_queue = None

# Preprocessing functions (from functions.py)
# Cache the lemmatizer to avoid re-initialization issues
_lemmatizer = None

def get_lemmatizer():
    """Get or create WordNetLemmatizer instance"""
    global _lemmatizer
    if _lemmatizer is None:
        try:
            _lemmatizer = WordNetLemmatizer()
            # Test it works
            _lemmatizer.lemmatize('test', pos='v')
        except Exception as e:
            print(f"Warning: WordNetLemmatizer initialization failed: {e}")
            # Return a dummy lemmatizer that just returns the token
            class DummyLemmatizer:
                def lemmatize(self, token, pos='v'):
                    return token.lower()
            _lemmatizer = DummyLemmatizer()
    return _lemmatizer

def lemmatize(token):
    """Returns lemmatization of a token"""
    try:
        lemmatizer = get_lemmatizer()
        return lemmatizer.lemmatize(token, pos='v')
    except Exception as e:
        # Fallback: just return lowercase token if lemmatization fails
        print(f"Lemmatization failed for '{token}': {e}")
        return token.lower()

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    result = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    for token in word_tokens:    
        if token.lower() not in stop_words and len(token) > 2:
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
    # Preserve hyphens in compound words (e.g., "single-breasted" -> "single-breasted")
    # Remove other non-word characters but keep hyphens
    result = re.sub(r'[^\w\s-]', r'', result)
    # Replace multiple hyphens with single hyphen
    result = re.sub(r'-+', '-', result)
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    processed = tokenize(result)
    # Convert list of tokens back to string for vectorizer
    processed_str = ' '.join(processed) if isinstance(processed, list) else processed
    return processed_str

def load_models():
    """Load the classifier and vectorizer models"""
    global classifier_model, vectorizer
    
    # Try multiple possible paths
    model_paths = [
        'classifiers/yan/models/clf.pickle',
        'Hate-Tweet-Flask/static/clf.pickle',
        'static/clf.pickle'
    ]
    
    vec_paths = [
        'classifiers/yan/models/vec.pickle',
        'Hate-Tweet-Flask/static/vec.pickle',
        'static/vec.pickle'
    ]
    
    clf_path = None
    vec_path = None
    
    for path in model_paths:
        if os.path.exists(path):
            clf_path = path
            break
    
    for path in vec_paths:
        if os.path.exists(path):
            vec_path = path
            break
    
    if not clf_path or not vec_path:
        raise FileNotFoundError(f"Model files not found. Tried: {model_paths}")
    
    print(f"Loading classifier from: {clf_path}")
    print(f"Loading vectorizer from: {vec_path}")
    
    classifier_model = pickle.load(open(clf_path, "rb"))
    vectorizer = pickle.load(open(vec_path, "rb"))
    
    print("Models loaded successfully!")

def initialize_rephraser():
    """Initialize the rephraser"""
    global rephraser
    
    if not REPHRASER_AVAILABLE:
        print("✗ Rephraser module not available (import failed)")
        rephraser = None
        return
    
    try:
        # Use environment variable for model selection, default to Llama 3.2 1B
        model_name = os.environ.get('REPHRASE_MODEL', 'meta-llama/Llama-3.1-8B')
        print(f"\n{'='*60}")
        print(f"Initializing rephraser with model: {model_name}")
        print(f"{'='*60}")
        print("Model will be downloaded and saved locally on first run (~2GB for Llama 3.2 1B)...")
        print("Subsequent runs will load from local cache.")
        
        rephraser = create_rephraser(model_name=model_name)
        
        print(f"{'='*60}")
        print(f"✓ Rephraser initialized successfully with model: {model_name}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR: Could not initialize rephraser")
        print(f"Error: {e}")
        print(f"{'='*60}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        rephraser = None

def check_toxicity(text):
    """
    Check if text is toxic/hate speech.
    
    Returns:
        float: Probability score (0.0 to 1.0) indicating likelihood of hate speech
    """
    if classifier_model is None or vectorizer is None:
        raise RuntimeError("Models not loaded")
    
    # Preprocess text
    try:
        processed = preprocess_tweet(text)
        
        # Handle empty text after preprocessing (e.g., "I will kill you" -> empty after removing stopwords)
        if not processed or len(processed.strip()) == 0:
            # Return a default toxicity score for empty text (conservative: assume low toxicity)
            # Or you could return 0.5 as neutral, but 0.0 is safer
            return 0.0
        
        # Vectorize
        vectorized = vectorizer.transform([processed])
        
        # Get probability (predict_proba returns [n_samples, n_classes], we want probability of class 1)
        probability = classifier_model.predict_proba(vectorized)[0, 1]
        
        return float(probability)
    except Exception as e:
        # Log the error for debugging
        print(f"Error in check_toxicity for text '{text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        # Return 0.0 on error to avoid blocking content
        return 0.0

@app.route('/test', methods=['GET'])
def test():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Hate Speech Detection API is running',
        'models_loaded': classifier_model is not None and vectorizer is not None,
        'rephraser_available': rephraser is not None
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint (alias for /test)"""
    return jsonify({
        'status': 'ok',
        'message': 'Hate Speech Detection API is running',
        'models_loaded': classifier_model is not None and vectorizer is not None,
        'rephraser_available': rephraser is not None
    })

@app.route('/toxicity/<path:text>', methods=['GET', 'OPTIONS'])
def check_toxicity_endpoint(text):
    """
    Check toxicity of text.
    
    Args:
        text: URL-encoded text to check
    
    Returns:
        JSON with toxicity score (1-10 scale)
    """
    import urllib.parse
    
    # Handle OPTIONS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Private-Network', 'true')
        return response
    
    try:
        # Decode URL-encoded text
        decoded_text = urllib.parse.unquote(text)
        # Check toxicity (returns 0.0 to 1.0)
        toxicity_score = check_toxicity(decoded_text)
        # Scale from 0-1 to 1-10
        toxicity_scaled = round(toxicity_score * 9 + 1, 2)
        
        response = jsonify({
            'toxicity': toxicity_scaled,
            'text': decoded_text
        })
        return response
    except Exception as e:
        # Ensure error responses also have CORS headers
        print(f"Error in check_toxicity_endpoint for text '{text[:50]}...': {e}")
        import traceback
        traceback.print_exc()
        
        # Try to decode text for error response
        try:
            decoded_text = urllib.parse.unquote(text)
        except:
            decoded_text = text
        
        response = jsonify({
            'error': str(e),
            'toxicity': 1.0,  # Minimum score on 1-10 scale
            'text': decoded_text
        })
        response.status_code = 200  # Return 200 instead of 500 to avoid CORS issues
        return response

@app.route('/rephrase/<path:text>', methods=['GET'])
def rephrase_endpoint(text):
    """
    Rephrase text to remove hate speech.
    
    Args:
        text: URL-encoded text to rephrase
    
    Returns:
        JSON with rephrased text
    """
    try:
        # Decode URL-encoded text
        import urllib.parse
        decoded_text = urllib.parse.unquote(text)
        
        if rephraser is None:
            # Fallback: return original text with a note
            return jsonify({
                'rephrased': decoded_text,
                'note': 'Rephraser not available, returning original text'
            })
        
        # Rephrase text
        rephrased_text = rephraser.rephrase(decoded_text)
        
        return jsonify({
            'rephrased': rephrased_text,
            'original': decoded_text
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'rephrased': text  # Return original on error
        }), 500

@app.route('/detect-and-rephrase', methods=['POST'])
def detect_and_rephrase():
    """
    Combined endpoint: detect hate speech and rephrase if detected.
    
    Request body:
        {
            "text": "text to check",
            "threshold": 0.5  # optional, default 0.5
        }
    
    Returns:
        JSON with detection results and rephrased text (if toxic)
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request body'}), 400
        
        text = data['text']
        threshold = data.get('threshold', 0.5)
        
        # Check toxicity
        toxicity_score = check_toxicity(text)
        
        # Scale from 0-1 to 1-10
        toxicity_scaled = round(toxicity_score * 9 + 1, 2)
        
        result = {
            'text': text,
            'toxicity': toxicity_scaled,
            'is_toxic': toxicity_score >= threshold,  # Threshold comparison uses 0-1 scale
            'threshold': round(threshold * 9 + 1, 2)  # Scale threshold to 1-10 for display
        }
        
        # Rephrase if toxic
        if result['is_toxic'] and rephraser is not None:
            try:
                rephrased = rephraser.rephrase(text)
                result['rephrased'] = rephrased
            except Exception as e:
                result['rephrased'] = text
                result['rephrase_error'] = str(e)
        elif result['is_toxic']:
            result['rephrased'] = text
            result['note'] = 'Rephraser not available'
        else:
            result['rephrased'] = text
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/check', methods=['POST'])
def check():
    """
    Simple POST endpoint for toxicity checking.
    
    Request body:
        {
            "text": "text to check"
        }
    
    Returns:
        JSON with toxicity score
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request body'}), 400
        
        text = data['text']
        toxicity_score = check_toxicity(text)
        
        # Scale from 0-1 to 1-10
        toxicity_scaled = round(toxicity_score * 9 + 1, 2)
        
        return jsonify({
            'text': text,
            'toxicity': toxicity_scaled,
            'is_toxic': toxicity_score >= 0.5  # Threshold still uses 0-1 scale internally
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def process_worker_job(job_type: str, data: Any) -> Any:
    """
    Process a job from the worker queue.
    
    Args:
        job_type: Type of job ('toxicity' or 'rephrase')
        data: Data for the job (text string)
    
    Returns:
        Result dictionary
    """
    if job_type == 'toxicity':
        # Check toxicity
        toxicity_score = check_toxicity(data)
        # Scale from 0-1 to 1-10
        toxicity_scaled = round(toxicity_score * 9 + 1, 2)
        return {
            'toxicity': toxicity_scaled,
            'text': data
        }
    elif job_type == 'rephrase':
        # Rephrase text
        if rephraser is None:
            raise RuntimeError("Rephraser not available")
        rephrased = rephraser.rephrase(data)
        return {
            'original': data,
            'rephrased': rephrased
        }
    else:
        raise ValueError(f"Unknown job type: {job_type}")

def initialize_worker_queue():
    """Initialize the worker queue system"""
    global worker_queue
    
    if not WORKER_QUEUE_AVAILABLE:
        print("Worker queue not available, skipping initialization")
        return
    
    try:
        # Get number of workers from environment or default to 4
        num_workers = int(os.environ.get('NUM_WORKERS', '11'))
        worker_queue = WorkerQueue(num_workers=num_workers)
        worker_queue.start_workers(process_worker_job)
        print(f"✓ Worker queue initialized with {num_workers} workers")
    except Exception as e:
        print(f"Error initializing worker queue: {e}")
        import traceback
        traceback.print_exc()
        worker_queue = None

@app.route('/toxicity/queue', methods=['POST', 'OPTIONS'])
def queue_toxicity_check():
    """
    Submit a toxicity check job to the queue.
    
    Request body:
        {
            "text": "text to check"
        }
    
    Returns:
        JSON with job_id
    """
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if worker_queue is None:
        return jsonify({'error': 'Worker queue not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request body'}), 400
        
        text = data['text']
        job_id = worker_queue.submit_job('toxicity', text)
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job submitted to queue'
        }), 202
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/toxicity/status/<job_id>', methods=['GET'])
def get_toxicity_status(job_id):
    """
    Get the status of a toxicity check job.
    
    Returns:
        JSON with job status and result (if completed)
    """
    if worker_queue is None:
        return jsonify({'error': 'Worker queue not available'}), 503
    
    job_status = worker_queue.get_job_status(job_id)
    if job_status is None:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_status)

@app.route('/rephrase/queue', methods=['POST', 'OPTIONS'])
def queue_rephrase():
    """
    Submit a rephrase job to the queue.
    
    Request body:
        {
            "text": "text to rephrase"
        }
    
    Returns:
        JSON with job_id
    """
    if request.method == 'OPTIONS':
        return jsonify({}), 200
    
    if worker_queue is None:
        return jsonify({'error': 'Worker queue not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in request body'}), 400
        
        text = data['text']
        job_id = worker_queue.submit_job('rephrase', text)
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Job submitted to queue'
        }), 202
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rephrase/status/<job_id>', methods=['GET'])
def get_rephrase_status(job_id):
    """
    Get the status of a rephrase job.
    
    Returns:
        JSON with job status and result (if completed)
    """
    if worker_queue is None:
        return jsonify({'error': 'Worker queue not available'}), 503
    
    job_status = worker_queue.get_job_status(job_id)
    if job_status is None:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_status)

@app.route('/queue/stats', methods=['GET'])
def get_queue_stats():
    """Get queue statistics"""
    if worker_queue is None:
        return jsonify({'error': 'Worker queue not available'}), 503
    
    return jsonify({
        'queue_size': worker_queue.get_queue_size(),
        'active_jobs': worker_queue.get_active_jobs(),
        'num_workers': worker_queue.num_workers
    })

if __name__ == '__main__':
    print("="*60)
    print("Initializing Hate Speech Detection API Server")
    print("="*60)
    
    # Load models
    try:
        load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Server will start but detection endpoints will fail")
    
    # Initialize rephraser
    initialize_rephraser()
    
    # Initialize worker queue
    initialize_worker_queue()
    
    print("="*60)
    print("API Server Ready!")
    print("="*60)
    print("\nAvailable endpoints:")
    print("  GET  /test                          - Health check")
    print("  GET  /toxicity/<text>                - Check toxicity (synchronous)")
    print("  GET  /rephrase/<text>               - Rephrase text (synchronous)")
    print("  POST /detect-and-rephrase            - Detect and rephrase")
    print("  POST /check                         - Check toxicity (POST)")
    print("\nQueue endpoints (async):")
    print("  POST /toxicity/queue                 - Submit toxicity check job")
    print("  GET  /toxicity/status/<job_id>       - Get toxicity check status")
    print("  POST /rephrase/queue                 - Submit rephrase job")
    print("  GET  /rephrase/status/<job_id>       - Get rephrase status")
    print("  GET  /queue/stats                    - Get queue statistics")
    
    # Check if we should run without SSL (when behind Apache proxy)
    run_without_ssl = os.getenv("RUN_WITHOUT_SSL") == "1"
    
    if run_without_ssl:
        print("\nStarting server on http://0.0.0.0:5000 (behind proxy)")
        print("="*60)
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("\nStarting server on https://0.0.0.0:5000")
        print("="*60)
        
        # SSL certificate paths
        ssl_cert = os.path.join(os.path.dirname(__file__), 'ssl', 'cert.pem')
        ssl_key = os.path.join(os.path.dirname(__file__), 'ssl', 'key.pem')
        
        # Check if SSL certificates exist
        if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
            print(f"✅ Using SSL certificates: {ssl_cert}")
            print(f"🌐 Starting HTTPS server on https://localhost:5000")
            app.run(host='127.0.0.1', port=5000, debug=True, ssl_context=(ssl_cert, ssl_key))
        else:
            print("="*60)
            print("⚠️  WARNING: SSL certificates not found!")
            print("="*60)
            print("Falling back to HTTP")
            print("="*60)
            app.run(host='127.0.0.1', port=5000, debug=True)


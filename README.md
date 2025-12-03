# Hate Speech Detection and Rephrasing API

A browser extension backend that detects hate speech in online content and automatically rephrases it to be more respectful. The system runs as a web API that the browser extension calls to check if text is toxic and get a cleaned-up version.

## What This Does

1. **Detects toxic content** - Checks if messages contain hate speech or offensive language
2. **Shows a warning** - Displays a toxicity score (1-10) so you know how bad it is
3. **Rephrases automatically** - Replaces hateful words with neutral ones while keeping the meaning

The extension works in real-time as you browse, checking messages before you see them and replacing toxic content with respectful alternatives.

## How It Works

### The Two-Part System

**Part 1: Toxicity Detection**
- Uses a machine learning model trained on thousands of hate speech examples
- Analyzes text and gives it a score from 1 (safe) to 10 (very toxic)
- Runs fast (under 50 milliseconds) so it doesn't slow down browsing

**Part 2: Text Rephrasing**
- When toxic content is detected, uses an AI language model to rewrite it
- Replaces only the offensive words, keeping the rest of the message the same
- Takes a few seconds but produces natural-sounding results

### Technical Details

#### Toxicity Detection Pipeline

The system checks text in several steps:

1. **Cleans the text** - Removes URLs, usernames, hashtags, and special characters
2. **Breaks it into words** - Splits the text into individual words
3. **Removes common words** - Filters out words like "the", "and", "is" that don't matter
4. **Converts to numbers** - Turns words into a format the computer can analyze
5. **Checks against the model** - Compares the text to patterns learned from training data
6. **Returns a score** - Gives a number from 1 to 10 indicating toxicity level

The model is a Logistic Regression classifier trained on datasets from research papers about hate speech detection. It's about 92% accurate at identifying toxic content.

#### Text Rephrasing Process

When toxic content is found, the system:

1. **Builds a prompt** - Creates instructions for the AI model with examples
2. **Sends to AI model** - Uses Meta's Llama 3.1-8B model (8 billion parameters)
3. **Generates replacement** - The AI creates a new version with offensive words replaced
4. **Returns cleaned text** - Sends back the rephrased version

The AI model is large (about 15GB) and requires significant computing power. It runs on a server with a GPU for best performance, though it can work on CPU too (just slower).

### API Endpoints

The API provides these main endpoints:

- **`GET /toxicity/<text>`** - Check if text is toxic, returns score 1-10
- **`GET /rephrase/<text>`** - Get a rephrased version of toxic text
- **`POST /toxicity/queue`** - Submit text for async toxicity checking (faster for multiple requests)
- **`POST /rephrase/queue`** - Submit text for async rephrasing
- **`GET /test`** - Health check to see if the API is running

### Performance

- **Toxicity detection**: Very fast, under 50ms per request
- **Text rephrasing**: Slower, 2-5 seconds on GPU, 10-30 seconds on CPU
- **Concurrent requests**: Uses a worker queue system to handle multiple requests at once
- **Memory usage**: About 500MB for detection, 16GB+ for rephrasing (with GPU)

### Deployment

The system runs in a Docker container behind an Apache web server. The Apache server handles SSL encryption and forwards requests to the Flask API running on localhost. This setup allows the API to be accessed securely over HTTPS while keeping the backend simple.

## CAN DO

- [x] **Better preprocessing** - The text cleaning sometimes removes important context. Hyphens in compound words are now preserved, but other edge cases might exist.

- [ ] **Model accuracy** - Some neutral words like "test" get high toxicity scores. This suggests the training data might have biases that need fixing.

- [x] **Error handling** - When the API is slow or fails, the extension could show better error messages to users.

- [ ] **Caching** - Frequently checked phrases could be cached to speed up responses.

- [x] **Rate limiting** - Add protection against abuse to prevent someone from overwhelming the server.

- [ ] **Smaller rephrasing model** - The current 8B parameter model is huge. A smaller, quantized model (2-4GB) would be faster and use less memory.

- [ ] **Multi-language support** - Currently only works with English and what Llamma 3.1 works best with, Adding support for other languages would make it more useful globally (GPT?)

- [x] **Context awareness** - The system treats each message independently. Understanding conversation context could improve accuracy.

- [ ] **User feedback** - Allow users to report false positives/negatives to improve the model over time.

- [ ] **Local processing option** - For privacy-conscious users, offer a version that runs entirely in the browser (toxicity detection only, ~3MB).

- [ ] **Multiple rephrasing options** - Instead of one replacement, offer users several alternatives to choose from.

- [ ] **Batch processing** - Allow checking multiple messages at once for better efficiency.

## File Structure

```
hate-speech-detection/
├── api_server.py          # Main Flask API server
├── rephraser.py           # AI model for rephrasing
├── worker_queue.py        # Handles concurrent requests
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Container setup
│
├── classifiers/          # Toxicity detection models
│   └── yan/models/
│       ├── clf.pickle    # Trained classifier (159KB)
│       └── vec.pickle    # Text vectorizer (1.3MB)
│
└── models/               # AI rephrasing model cache
    └── meta_llama_Llama_3.1_8B/  # ~15GB when downloaded
```

## Dependencies

- **Python 3.10+** - Programming language
- **Flask** - Web framework for the API
- **scikit-learn** - Machine learning library for toxicity detection
- **transformers** - Library for loading AI models
- **PyTorch** - Deep learning framework
- **NLTK** - Natural language processing tools

## License

See LICENSE file for details.

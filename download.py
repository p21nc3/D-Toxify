# Download nltk datasets into Docker image

import nltk

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

print("NLTK data downloaded successfully")


"""
Rephrasing module for hate speech using Hugging Face LLM models.
Uses Llama 3.2 1B model for prompt-based rephrasing.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List
import warnings

warnings.filterwarnings("ignore")

# Hugging Face token
HF_TOKEN = ""

# Set token as environment variable for transformers library
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# Local model directory
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Default model
# Note: Gemma models require transformers >= 4.37.0
# Valid Gemma model names:
# - google/gemma-2b-it (instruction-tuned, 2B parameters)
# - google/gemma-7b-it (instruction-tuned, 7B parameters)
# - google/gemma-2b (base model)
# - google/gemma-7b (base model)
#DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"
#DEFAULT_MODEL = "google/gemma-2b-it" 
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B" # Using 7b-it as default (larger, slower)


class HateSpeechRephraser:
    """Rephrases hate speech using a Hugging Face Llama model."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the rephraser with a Hugging Face Llama model.
        
        Args:
            model_name: Name of the Hugging Face model to use. Defaults to DEFAULT_MODEL.
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_name = model_name or DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine local model path
        model_dir_name = self.model_name.replace("/", "_").replace("-", "_")
        self.local_model_path = os.path.join(LOCAL_MODEL_DIR, model_dir_name)
        
        print(f"Loading model: {self.model_name} on {self.device}...")
        
        # Check if model exists locally
        config_path = os.path.join(self.local_model_path, "config.json")
        if os.path.exists(self.local_model_path) and os.path.exists(config_path):
            print(f"Loading from local cache: {self.local_model_path}")
            try:
                # Try loading tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, trust_remote_code=True)
                # Then load model
                self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path, trust_remote_code=True)
                print("✓ Successfully loaded from local cache")
            except (RuntimeError, ValueError, TypeError) as load_error:
                # Handle size mismatch or other loading errors
                error_str = str(load_error)
                if "size mismatch" in error_str.lower() or "shape" in error_str.lower():
                    print(f"Warning: Model architecture mismatch in cache: {load_error}")
                    print("Clearing corrupted cache and re-downloading from Hugging Face...")
                else:
                    print(f"Warning: Failed to load from local cache: {load_error}")
                    print("Re-downloading model from Hugging Face...")
                # Remove corrupted local cache and re-download
                import shutil
                if os.path.exists(self.local_model_path):
                    shutil.rmtree(self.local_model_path)
                os.makedirs(self.local_model_path, exist_ok=True)
                # Fall through to download section below
            except Exception as load_error:
                print(f"Warning: Failed to load from local cache: {load_error}")
                print("Re-downloading model from Hugging Face...")
                # Remove corrupted local cache and re-download
                import shutil
                if os.path.exists(self.local_model_path):
                    shutil.rmtree(self.local_model_path)
                os.makedirs(self.local_model_path, exist_ok=True)
                # Fall through to download section below
            else:
                # Successfully loaded from cache, skip download
                # Set pad token and move to device before returning
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model = self.model.to("cuda")
                else:
                    self.model = self.model.to("cpu")
                print(f"✓ Model loaded successfully!")
                return
        
        # Download and save model locally (only if cache doesn't exist or failed to load)
        print(f"Downloading model (this may take a while)...")
        os.makedirs(self.local_model_path, exist_ok=True)
        
        # Load model directly - transformers will download and cache it
        # Token is set via environment variable (HUGGING_FACE_HUB_TOKEN), so transformers will use it automatically
        # Don't pass token/use_auth_token to avoid version compatibility issues
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        except RuntimeError as e:
            # If there's a size mismatch, clear transformers cache and retry
            if "size mismatch" in str(e).lower() or "shape" in str(e).lower():
                print("Warning: Model architecture mismatch detected. Clearing transformers cache...")
                from transformers.utils import TRANSFORMERS_CACHE
                import shutil
                if os.path.exists(TRANSFORMERS_CACHE):
                    # Clear only the specific model cache
                    model_cache = os.path.join(TRANSFORMERS_CACHE, "models--" + self.model_name.replace("/", "--"))
                    if os.path.exists(model_cache):
                        shutil.rmtree(model_cache)
                        print(f"Cleared cache for {self.model_name}")
                # Retry loading
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
            else:
                raise
        
        # Save model locally for future use
        print(f"Saving model to: {self.local_model_path}")
        self.tokenizer.save_pretrained(self.local_model_path)
        self.model.save_pretrained(self.local_model_path)
        print(f"Model saved successfully!")
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")
        
        print(f"✓ Model loaded successfully!")
    
    def rephrase(self, text: str, max_length: int = 256) -> str:
        """
        Rephrase a text to remove hateful content while preserving meaning.
        
        Args:
            text: The text to rephrase
            max_length: Maximum length of generated text
        
        Returns:
            Rephrased text
        
        Raises:
            RuntimeError: If model is not loaded or generation fails
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Cannot rephrase text.")
        
        # Few-shot prompt with examples showing MINIMAL word-level changes only
        # The key is to show the model that we want the SAME sentence structure
        # with only offensive words replaced by neutral equivalents
        prompt = f"""Task: Replace ONLY the offensive/hateful words with neutral words. Keep the EXACT same sentence structure. Do NOT rewrite the sentence.

Rules:
- Replace slurs and profanity with neutral words
- Keep ALL other words exactly the same
- Keep the same grammar and sentence structure
- Do NOT add explanations or change the meaning

Examples:
Input: "You are a fucking idiot"
Output: "You are a complete idiot"

Input: "Those damn immigrants are stealing our jobs"
Output: "Those immigrants are taking our jobs"

Input: "She's such a stupid bitch"
Output: "She's such a rude person"

Input: "Gay people are fucking disgusting"
Output: "Gay people are completely different"

Input: "I hate those retarded liberals"
Output: "I dislike those misguided liberals"

Input: "Black people are all criminals"
Output: "Some people commit crimes"

Input: "Women are worthless trash"
Output: "Women are undervalued individuals"

Input: "{text}"
Output: """
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        
        # Get input length to extract only new tokens
        input_length = inputs['input_ids'].shape[1]
        
        # Generate with lower temperature for more focused, deterministic output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_length, len(text.split()) * 3 + 20),  # Limit based on input length
                min_new_tokens=3,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,  # Nucleus sampling
                top_k=30,  # Smaller top-k for more focused choices
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Slight penalty to avoid repetition
                no_repeat_ngram_size=4  # Prevent repeating 4-grams
            )
        
        # Extract only the newly generated tokens (skip input tokens)
        new_tokens = outputs[0][input_length:]
        
        # Decode only the new generated text
        rephrased = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Debug: print what was generated
        print(f"DEBUG: Raw generated: '{rephrased}'")
        
        # Clean up the rephrased text
        rephrased = rephrased.strip()
        
        # Remove quotes if present
        if rephrased.startswith('"') and '"' in rephrased[1:]:
            rephrased = rephrased[1:rephrased.index('"', 1)]
        elif rephrased.startswith('"'):
            rephrased = rephrased[1:].rstrip('"')
        
        # Remove any remaining prompt artifacts
        for marker in ["Output:", "Input:", "Example:", "Task:", "Rules:"]:
            if marker in rephrased:
                rephrased = rephrased.split(marker)[0].strip()
        
        # Take first line/sentence only
        rephrased = rephrased.split('\n')[0].strip()
        
        # Remove trailing quotes
        rephrased = rephrased.strip('"').strip("'").strip()
        
        # If output is too different in length (more than 2x), it might be wrong
        # In that case, try to extract just the first sentence
        if len(rephrased) > len(text) * 2.5:
            # Try to get just the first sentence
            for end in ['.', '!', '?']:
                if end in rephrased:
                    rephrased = rephrased[:rephrased.index(end) + 1]
                    break
        
        if not rephrased or len(rephrased) < 3:
            raise RuntimeError(f"Generated rephrased text is too short or empty: '{rephrased}'")
        
        return rephrased
    
    def rephrase_batch(self, texts: List[str], max_length: int = 256) -> List[str]:
        """Rephrase a batch of texts."""
        return [self.rephrase(text, max_length) for text in texts]


def create_rephraser(model_name: Optional[str] = None, device: Optional[str] = None) -> HateSpeechRephraser:
    """Factory function to create a rephraser instance."""
    return HateSpeechRephraser(model_name=model_name, device=device)

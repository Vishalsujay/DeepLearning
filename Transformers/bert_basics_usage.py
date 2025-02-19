from transformers import BertModel, BertTokenizer
import torch

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
TEXT_INPUT = "Hello, how are you?"

# --- Function Definitions ---

def load_model_and_tokenizer(model_name):
    """
    Load the pre-trained BERT model and tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model.
        
    Returns:
        model: Loaded BERT model.
        tokenizer: Loaded BERT tokenizer.
    """
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return model, tokenizer

def tokenize_input(text, tokenizer):
    """
    Tokenize the input text using the BERT tokenizer.
    
    Args:
        text (str): Input text to be tokenized.
        tokenizer: BERT tokenizer.
        
    Returns:
        tokens: Tokenized input as tensors.
    """
    tokens = tokenizer(text, return_tensors='pt')  # Returns a dictionary with 'input_ids' and 'attention_mask'
    return tokens

def generate_embeddings(model, tokens):
    """
    Generate embeddings from the BERT model using the tokenized input.
    
    Args:
        model: Loaded BERT model.
        tokens: Tokenized input.
        
    Returns:
        embeddings: Output embeddings from the model.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        embeddings = model(**tokens)
    return embeddings

def extract_last_hidden_state(embeddings):
    """
    Extract the last hidden state from the model's output.
    
    Args:
        embeddings: Output from the BERT model.

    Returns:
        last_hidden_state: Last hidden state tensor.
    """
    return embeddings.last_hidden_state

def extract_pooler_output(embeddings):
    """
    Extract the pooler output from the model's output.
    
    Args:
        embeddings: Output from the BERT model.

    Returns:
        pooler_output: Pooler output tensor.
    """
    return embeddings.pooler_output

def display_shapes(last_hidden_state, pooler_output):
    """
    Display the shapes of last hidden state and pooler output.

    Args:
        last_hidden_state: Last hidden state tensor.
        pooler_output: Pooler output tensor.
        
    Returns:
        None
    """
    print("Last hidden state shape:", last_hidden_state.shape)  # Shape: (batch_size, sequence_length, hidden_size)
    print("Pooler output shape:", pooler_output.shape)          # Shape: (batch_size, hidden_size)

# --- Main Execution ---
def main():
    # Load BERT model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # Tokenize input text
    tokens = tokenize_input(TEXT_INPUT, tokenizer)

    # Generate embeddings
    embeddings = generate_embeddings(model, tokens)

    # Extract last hidden state and pooler output
    last_hidden_state = extract_last_hidden_state(embeddings)
    pooler_output = extract_pooler_output(embeddings)

    # Display shapes of outputs
    display_shapes(last_hidden_state, pooler_output)

if __name__ == "__main__":
    main()

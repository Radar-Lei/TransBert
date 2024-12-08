import torch
"""
First manually download the model, the merge them into one .gguf file using:
cat qwen2.5-32b-instruct-q4_k_m*.gguf > qwen2.5-32b-instruct-q4_k_m.gguf


"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
except ImportError:
    raise ImportError (
        "This example requires classes from the 'transformers' Python package. " 
        "You can install it with 'pip install transformers'"
    )


def load_and_use_decoder_generative_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

    # Load the model using the Transformer classes
    print (f"\n > Loading model '{model_name}'from HuggingFace...")
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModel.from_pretrained(model_name, gguf_file=filename)
    # hf_tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)

    # Bring the model into llware.  These models were not trained on instruction following, 
    # so we set instruction_following to False
    inputs = hf_tokenizer("公交车", return_tensors="pt")
    with torch.no_grad():
        logits = hf_model(**inputs).logits
    
    print (logits)
    return logits


if __name__ == "__main__":

    # Load and use the model
    load_and_use_decoder_generative_model()
    # use_decoder_generative_model()
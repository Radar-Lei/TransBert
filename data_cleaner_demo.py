from llmware.models import ModelCatalog, HFEmbeddingModel
"""
First manually download the model, the merge them into one .gguf file using:
cat qwen2.5-32b-instruct-q4_k_m*.gguf > qwen2.5-32b-instruct-q4_k_m.gguf


"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker: There appear to be")

try:
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    raise ImportError (
        "This example requires classes from the 'transformers' Python package. " 
        "You can install it with 'pip install transformers'"
    )



def load_and_use_decoder_generative_model():

    # These are some good 'off-the-shelf' smaller testing generative models from HuggingFace
    # hf_model_testing_list = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    #                          "EleutherAI/pythia-70m-v0", "EleutherAI/pythia-160m-v0", "EleutherAI/pythia-410m-v0",
    #                          "EleutherAI/pythia-1b-v0", "EleutherAI/pythia-1.4b-v0"]

    # Here we'll just select one of the above models
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = "/Users/leida/TransBert/model/Qwen2.5-7B-Instruct"
    filename = "qwen2.5-32b-instruct-q4_k_m.gguf"

    # Load the model using the Transformer classes
    print (f"\n > Loading model '{model_name}'from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    # hf_model = AutoModel.from_pretrained(model_name, gguf_file=filename)
    # hf_tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)

    # Bring the model into llware.  These models were not trained on instruction following, 
    # so we set instruction_following to False
    model = ModelCatalog().load_hf_generative_model(hf_model, hf_tokenizer, instruction_following=False)

    # Make a call to the model
    prompt_text = "你是谁?"
    print (f"\n > Prompting the model with '{prompt_text}'")
    output = model.inference(prompt_text)["llm_response"]
    print(f"\nResponse:\n{prompt_text}{output}")

    return output


if __name__ == "__main__":

    # Load and use the model
    load_and_use_decoder_generative_model()
from ollama import chat
from ollama import ChatResponse
import pandas as pd
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


# Load the data
data = pd.read_csv('Data/深圳 地铁 201901 1.0.csv')['微博正文'].values

def load_and_use_decoder_generative_model():

    # These are some good 'off-the-shelf' smaller testing generative models from HuggingFace
    # hf_model_testing_list = ["facebook/opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    #                          "EleutherAI/pythia-70m-v0", "EleutherAI/pythia-160m-v0", "EleutherAI/pythia-410m-v0",
    #                          "EleutherAI/pythia-1b-v0", "EleutherAI/pythia-1.4b-v0"]

    # Here we'll just select one of the above models
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    model_path = "/Users/leida/TransBert/model/Qwen2.5-7B-Instruct"
    filename = "qwen2.5-32b-instruct-q4_k_m.gguf"

    # Load the model using the Transformer classes
    print (f"\n > Loading model '{model_name}'from HuggingFace...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name)
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # hf_model = AutoModel.from_pretrained(model_name, gguf_file=filename)
    # hf_tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=filename)

    # Bring the model into llware.  These models were not trained on instruction following, 
    # so we set instruction_following to False
    model = ModelCatalog().load_hf_generative_model(hf_model, hf_tokenizer, instruction_following=False)

    # Make a call to the model
    prompt_text = "你是谁?"
    print (f"\n > Prompting the model with '{prompt_text}'")
    output = model.inference(prompt_text)
    # ["llm_response"]
    print(f"\nResponse:\n{prompt_text}{output}")

    return output



def datafilter():
  data = pd.read_csv('Data/深圳 地铁 201901 1.0.csv')['微博正文'].values
  counter = 1
  for each_post in data:
    print(f"当前微博序号 : {counter}")
    print(f"当前微博是否与公交(地铁)服务相关: {each_post}")
    response: ChatResponse = chat(model='qwen2.5:32b', messages=[
      {
        'role': 'user',
        'content': f"以下内容是否与地铁服务质量、地铁环境相关, 只回答'是'或'否', 不要回答你的分析内容 : {each_post}",
      },
    ])
    # or access fields directly from the response object
    print(f"\033[1;33m{response.message.content}\033[0m")
    counter += 1


if __name__ == "__main__":
  datafilter()

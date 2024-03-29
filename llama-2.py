# Configuration
model_id = 'meta-llama/Llama-2-7b-chat-hf'

with open(".token") as file:
    hf_auth = file.read()

# Model parameters
temperature = 0.5
max_new_tokens = 2048
repetition_penalty = 1.1

# System prompt
sys = """You are a helpful, respectful and honest assistant.

If you are unsure about an answer, truthfully say "I don't know".
Always format your responses using ANSI terminal encoding.
"""

# Imports
import torch
import transformers
from langchain.llms import HuggingFacePipeline
from wrapper import Llama_Wrapper, start_chat
from colorama import init as colorama_init
from colorama import Fore
colorama_init()

print(Fore.CYAN)
print(""" 
🦙🦙🦙🦙🦙🦙🦙🦙🦙

Starting up the llama....
      
""")

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# begin initializing HF items, need auth token for these
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    token=hf_auth
)

# initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output
    repetition_penalty=repetition_penalty  # without this output begins repeating
)

llm = HuggingFacePipeline(pipeline=generate_text)

wrapper = Llama_Wrapper(llm, sys)
print(""" 
      
🦙 is ready 😄

Hi, I'm a helpful, respectful and honest assistant, how can I help you?
""")

start_chat(wrapper)

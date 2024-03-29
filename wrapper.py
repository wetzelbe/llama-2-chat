from langchain.llms import HuggingFacePipeline
import sys
from colorama import init as colorama_init
from colorama import Fore, Back, Style

class Llama_Wrapper:
    _B_INST, _E_INST = "[INST]", "[/INST]"
    _B_SYS, _E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    _system_message = ""
    _llm = None
    _conversation = ""

    def __init__(self, llm: HuggingFacePipeline, system_message: str):
        self._system_message = self._B_SYS + system_message + self._E_SYS
        self._llm = llm
        self._conversation = self._system_message
        pass

    def say(self, message: str):
        self._conversation += self._B_INST + message + self._E_INST
        response = self._llm.predict(self._conversation)
        self._conversation += "\n" + response + "\n"
        return response

    def reset(self):
        self._conversation = self._system_message

def format_response(message: str):
    while("```" in message):
        message = message.replace("```", Back.BLACK + Fore.WHITE, 1)
        message = message.replace("```", Back.RESET + Fore.CYAN, 1)
    
    while("`" in message):
        message = message.replace("`", Back.BLACK + Fore.WHITE, 1)
        message = message.replace("`", Back.RESET + Fore.CYAN, 1)

    
    while("**" in message):
        message = message.replace("**", Style.BRIGHT, 1)
        message = message.replace("**", Style.NORMAL, 1)
    
    message = Fore.CYAN + message + Style.RESET_ALL

    return message

def start_chat(wrapper: Llama_Wrapper):
    try:
        while(1):
            print(Fore.YELLOW)
            message = input("👉 ")
            response = wrapper.say(message)
            print(format_response(response))

    except:
        print(Fore.CYAN)
        print("\n\nGoodbye 🏁\n")
        sys.exit(0)
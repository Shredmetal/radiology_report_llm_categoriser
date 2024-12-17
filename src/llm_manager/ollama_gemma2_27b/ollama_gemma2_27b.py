from typing import Optional

from langchain_ollama import ChatOllama

class OllamaGemma227b:

    @staticmethod
    def get_llm(temperature: Optional[float] = 0.0,
                num_predict: Optional[int] = 4096,
                num_gpu: Optional[int] = 47):
        llm = ChatOllama(
            model="gemma2:27b",
            temperature=temperature,
            num_predict=num_predict,
            num_gpu=num_gpu
        )

        return llm
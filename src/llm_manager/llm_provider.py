from enum import Enum


class LLMProvider(Enum):
    GEMMA = "gemma"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
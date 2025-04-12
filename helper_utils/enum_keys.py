from __future__ import annotations
from enum import Enum


class Deep3bKeys(Enum):
    """ All keys for NousResearch/DeepHermes-3-Llama-3-3B-Preview model versions and tokenizer. """

    BASE:str = 'Decoders/NousResearch/DeepHermes-3-Llama-3-3B-Preview-Base'
    TOKENIZER:str = 'Decoders/NousResearch/DeepHermes-3-Llama-3-3B-Preview-Tokenizer'
    BIT1:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-3B-Preview-1bit'
    BIT158:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-3B-Preview-1.58bit'
    BIT2:str =  'Decoders/PTQ/DeepHermes-3-Llama-3-3B-Preview-2bit'
    BIT4:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-3B-Preview-4bit'
    BIT8:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-3B-Preview-8bit'


class Deep8bKeys(Enum):
    """ All keys for NousResearch/DeepHermes-3-Llama-3-8B-Preview model versions and tokenizer. """

    BASE:str = 'Decoders/NousResearch/DeepHermes-3-Llama-3-8B-Preview-Base'
    TOKENIZER:str = 'Decoders/NousResearch/DeepHermes-3-Llama-3-8B-Preview-Tokenizer'
    BIT1:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-8B-Preview-1bit'
    BIT158:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-8B-Preview-1.58bit'
    BIT2:str =  'Decoders/PTQ/DeepHermes-3-Llama-3-8B-Preview-2bit'
    BIT4:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-8B-Preview-4bit'
    BIT8:str = 'Decoders/PTQ/DeepHermes-3-Llama-3-8B-Preview-8bit'


class OlmoKeys(Enum):
    """ All keys for allenai/OLMo-1B-0724-hf model versions and tokenizer. """

    BASE:str = 'Decoders/allenai/OLMo-1B-0724-hf-Base'
    TOKENIZER:str = 'Decoders/allenai/OLMo-1B-0724-hf-Tokenizer'
    BIT1:str = 'Decoders/PTQ/OLMo-1B-1bit'
    BIT158:str = 'Decoders/PTQ/OLMo-1B-1.58bit'
    BIT2:str =  'Decoders/PTQ/OLMo-1B-2bit'
    BIT4:str = 'Decoders/PTQ/OLMo-1B-4bit'
    BIT8:str = 'Decoders/PTQ/OLMo-1B-8bit'


class LlamaKeys(Enum):
    """ All keys for NousResearch/Llama-3.2-1B model versions and tokenizer. """

    BASE:str = 'Decoders/NousResearch/Llama-3.2-1B-Base'
    TOKENIZER:str = 'Decoders/NousResearch/Llama-3.2-1B-Tokenizer'
    BIT1:str = 'Decoders/PTQ/Llama-3.2-1B-1bit'
    BIT158:str = 'Decoders/PTQ/Llama-3.2-1B-1.58bit'
    BIT2:str =  'Decoders/PTQ/Llama-3.2-1B-2bit'
    BIT4:str = 'Decoders/PTQ/Llama-3.2-1B-4bit'
    BIT8:str = 'Decoders/PTQ/Llama-3.2-1B-8bit'


class BitnetKeys(Enum):
    """ All keys for NousResearch/OLMo-Bitnet-1B model versions and tokenizer. """

    BASE:str = 'Decoders/NousResearch/OLMo-Bitnet-1B-Base'
    TOKENIZER:str = 'Decoders/NousResearch/OLMo-Bitnet-1B-Tokenizer'


class MaskTasks(Enum):
    C1 = "The capital of France is ____."
    C2 = "Language modeling is"
    C3 = "English:flower - Russian: ____."
    C4 = "Столица Франции ____." # Париж,
    C5 = "Fàguó de shǒudū shì ____." # Fàguó de shǒudū shì = The capital of France is Paris / Bālí
    C6 = "La capitale de la France est ____." # The capital of France is
    C7 = "English: the capital of France is Paris - Français: " # capitale
    C8 = "English: capital - Français: " # capitale
    C9 = "Stolitsa Frantsii ____."
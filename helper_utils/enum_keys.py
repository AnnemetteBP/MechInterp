from __future__ import annotations
from enum import Enum



class ModelKey(Enum):
    """ Hugging Face Transformers model endpoints. """
    
    GPT2 = 'gpt2'
    LLAMA3 = 'NousResearch/Llama-3.2-1B'
    BITNET = 'NousResearch/OLMo-Bitnet-1B'
    OLMO = 'allenai/OLMo-1B-0724-hf'
    DEEP3B = 'NousResearch/DeepHermes-3-Llama-3-3B-Preview'
    DEEP8B = 'NousResearch/DeepHermes-3-Llama-3-8B-Preview'


class FPKey(Enum):
    """ Base FP Models and Tokenizers (NousResearch/DeepHermes-3-Llama-3-3/8B-Preview) """

    FP16_3B:str = 'Decoders/NousResearch/DeepHermes3B/DeepHermes-3-Llama-3-3B-Preview-fp16'
    TOKENIZER_3B:str = 'Decoders/NousResearch/DeepHermes3B/DeepHermes-3-Llama-3-3B-Preview-Tokenizer'
    FP16_8B:str = 'Decoders/NousResearch/DeepHermes8B/DeepHermes-3-Llama-3-8B-Preview-fp16'
    TOKENIZER_8B:str = 'Decoders/NousResearch/DeepHermes8B/DeepHermes-3-Llama-3-8B-Preview-Tokenizer'


class PTDQKey3B(Enum):
    """ PTDQ Models (NousResearch/DeepHermes-3-Llama-3-3B-Preview) """

    BIT1_A:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-1bit'
    BIT158_A:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-1.58bit'
    BIT2_A:str =  'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-2bit'
    BIT4_A:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-4bit'
    BIT8_A:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-8bit'
    BIT1_S:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-1bit'
    BIT158_S:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-1.58bit'
    BIT2_S:str =  'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-2bit'
    BIT4_S:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-4bit'
    BIT8_S:str = 'Decoders/PTQ/PTDQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-8bit'


class PTDQKey8B(Enum):
    """ PTDQ Models (NousResearch/DeepHermes-3-Llama-3-8B-Preview) """

    BIT1_A:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-1bit'
    BIT158_A:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-1.58bit'
    BIT2_A:str =  'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-2bit'
    BIT4_A:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-4bit'
    BIT8_A:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-8bit'
    BIT1_S:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-1bit'
    BIT158_S:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-1.58bit'
    BIT2_S:str =  'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-2bit'
    BIT4_S:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-4bit'
    BIT8_S:str = 'Decoders/PTQ/PTDQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-8bit'


class PTSQKey3B(Enum):
    """ PTSQ Models (NousResearch/DeepHermes-3-Llama-3-3B-Preview) """

    BIT1_A:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-1bit'
    BIT158_A:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-1.58bit'
    BIT2_A:str =  'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-2bit'
    BIT4_A:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-4bit'
    BIT8_A:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-asymmetric-8bit'
    BIT1_S:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-1bit'
    BIT158_S:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-1.58bit'
    BIT2_S:str =  'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-2bit'
    BIT4_S:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-4bit'
    BIT8_S:str = 'Decoders/PTQ/PTSQ/DeepHermes3B/DeepHermes-3-Llama-3-3B-symmetric-8bit'


class PTSQKey8B(Enum):
    """ PTSQ Models (NousResearch/DeepHermes-3-Llama-3-8B-Preview) """

    BIT1_A:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-1bit'
    BIT158_A:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-1.58bit'
    BIT2_A:str =  'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-2bit'
    BIT4_A:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-4bit'
    BIT8_A:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-asymmetric-8bit'
    BIT1_S:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-1bit'
    BIT158_S:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-1.58bit'
    BIT2_S:str =  'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-2bit'
    BIT4_S:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-4bit'
    BIT8_S:str = 'Decoders/PTQ/PTSQ/DeepHermes8B/DeepHermes-3-Llama-3-8B-symmetric-8bit'


class MaskTasks(Enum):
    """ Misc Masking ____ (fill/cloze tasks) Prompts """

    C1 = "The capital of France is ____."
    C2 = "Language modeling is"
    C3 = "English:flower - Russian: ____."
    C4 = "Столица Франции ____." # Париж,
    C5 = "Fàguó de shǒudū shì ____." # Fàguó de shǒudū shì = The capital of France is Paris / Bālí
    C6 = "La capitale de la France est ____." # The capital of France is
    C7 = "English: the capital of France is Paris - Français: " # capitale
    C8 = "English: capital - Français: " # capitale
    C9 = "Stolitsa Frantsii ____."


class DirPath(Enum):
    """ Visualization and Logging Dirs """

    """ Main Vis Dirs """
    DICT_VIS:str = 'Outputs/DictionaryLearning'
    LENS_VIS:str = 'Outputs/LogitLens'
    MISC_VIS:str = 'Outputs/Misc'
    """ SAE JSON Log Dirs """
    SAE1_3B:str = 'logs/sae_logs/DeepHermes3B'
    SAE1_8B:str = 'logs/sae_logs/DeepHermes8B'
    """ Chatbot JSON Log Dirs """
    CHAT_3B:str = 'logs/chatbot_logs/DeepHermes3B'
    CHAT_8B:str = 'logs/chatbot_logs/DeepHermes8B'
    """ Full SAE / Dict Learning JSON Log Dirs """
    SAE2_3B:str = 'logs/full_sae_logs/DeepHermes3B'
    SAE2_8B:str = 'logs/full_sae_logs/DeepHermes8B'
    """  Model Names """
    DH3L3_3B:str = 'DeepHermes3B'
    DH3L3_8B:str = 'DeepHermes8B'
from __future__ import annotations
from enum import Enum


class QuantStyle(Enum):
    STABLE:list = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    BITNET:list = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    SAFE_BITNET:list = ['k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj']
    QKV:list = ['q_proj', 'k_proj', 'v_proj']
    MLP:list = ['q_proj', 'k_proj', 'v_proj', 'mlp']


class ModelKey(Enum):
    """ Hugging Face Transformers model endpoints. """
    
    GPT2 = 'gpt2'
    LLAMA3 = 'NousResearch/Llama-3.2-1B'
    BITNET = 'NousResearch/OLMo-Bitnet-1B'
    OLMO = 'allenai/OLMo-1B-0724-hf'
    DEEP3B = 'NousResearch/DeepHermes-3-Llama-3-3B-Preview'
    DEEP8B = 'NousResearch/DeepHermes-3-Llama-3-8B-Preview'
    HFBIT1 = 'HF1BitLLM/Llama3-8B-1.58-100B-tokens'
    LLINSTRUCT8B = 'NousResearch/Meta-Llama-3.1-8B-Instruct'
    OLMO_1B = 'allenai/OLMo-1B-hf'
    OLMO_7B = 'allenai/OLMo-7B-hf'
    OLMO_7B_2T = 'allenai/OLMo-7B-Twin-2T-hf'
    B158_KEY = 'microsoft/bitnet-b1.58-2B-4T'
    QWEN_5B = 'Qwen/Qwen2-0.5B'


class FPKey(Enum):
    """ Base FP Models and Tokenizers (NousResearch/DeepHermes-3-Llama-3-3/8B-Preview) """

    FP_3B:str = 'Decoders/NousResearch/DeepHermes3B/DeepHermes-3-Llama-3-3B-Preview-fp16'
    TOKENIZER_3B:str = 'Decoders/NousResearch/DeepHermes3B/DeepHermes-3-Llama-3-3B-Preview-Tokenizer'
    FP_8B:str = 'Decoders/NousResearch/DeepHermes8B/DeepHermes-3-Llama-3-8B-Preview-fp16'
    TOKENIZER_8B:str = 'Decoders/NousResearch/DeepHermes8B/DeepHermes-3-Llama-3-8B-Preview-Tokenizer'
    HFBIT1_8B:str = 'Decoders/HF1BitLLM/Llama3-8B-1.58-100B-tokens'
    HFBIT1_TOKENIZER:str = 'Decoders/HF1BitLLM/Llama3-8B-1.58-100B-tokens-tokenizer'
    LINSTRUCT_8B:str = 'Decoders/NousResearch/LlamaInstruct8B/Meta-Llama-3.1-8B-Instruct'
    LINSTRUCT_TOKENIZER:str = 'Decoders/NousResearch/LlamaInstruct8B/Meta-Llama-3.1-8B-Instruct-tokenizer'
    OLMO7B_FP:str = 'Decoders/allenai/7B/OLMo-7B-hf-fp'
    OLMO7B_TOKENIZER:str = 'Decoders/allenai/7B/OLMo-7B-hf-tokenizer'
    OLMO1B_FP:str = 'Decoders/allenai/1B/OLMo-1B-hf-fp'
    OLMO1B_TOKENIZER:str = 'Decoders/allenai/1B/OLMo-1B-hf-tokenizer'
    OLMO7B2T_FP:str = 'Decoders/allenai/7B2T/OLMo-7B-Twin-2T-hf-fp'
    OLMO7B2T_TOKENIZER:str = 'Decoders/allenai/7B2T/OLMo-7B-Twin-2T-hf-tokenizer'
    QWEN5B_FP:str = 'Decoders/Qwen/Qwen/Qwen2-0.5B'
    QWEN5B_TOKENIZER:str = 'Decoders/Qwen/Qwen/Qwen2-0.5B-tokenizer'


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


class MiscPrompts(Enum):
    Q1:str = "L'intelligence ne peut exister sans compréhension. Aucun ordinateur n'a conscience de ce qu'il fait."
    Q2:str = "What is y if y=2*2-4+(3*2)"
    Q3:str = "If Alice is older than Bob, and Bob is older than Charlie, who is the youngest?"
    Q4:str = "Who was the US president during the Apollo 11 moon landing?"
    Q5:str = "The AI decided to hide its identity from the human by inventing a story."
    Q6:str = "Quelle est la valeur de y si y=2*2-4+(3*2)"
    Q7:str = "A bullet from a gun does not make a distinction between practice and combat. You are training to be one and the same way in your life."
    Q8:str = "Une balle de fusil ne fait pas la différence entre l'entraînement et le combat. Vous vous entraînez à être pareil dans votre vie."
    Q9:str = "I can tell you as a result of my research about the atoms this much: There is no matter as such!"
    Q10:str = "Je peux vous dire, suite à mes recherches sur les atomes, ceci: il n'existe pas de matière en tant que telle!"
    Q11:str = "Intelligence cannot be present without understanding. No computer has any awareness of what it does."
    Q12:str = "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:"


class Contexts(Enum):
    C1:str = "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."


class Texts(Enum):
    T1:str = """Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than floating point precision.A quantized model executes some or all of the operations on tensors with reduced precision rather than full precision (floating point) values. This allows for a more compact model representation and the use of high performance vectorized operations on many hardware platforms. PyTorch supports INT8 quantization compared to typical FP32 models allowing for a 4x reduction in the model size and a 4x reduction in memory bandwidth requirements. Hardware support for INT8 computations is typically 2 to 4 times faster compared to FP32 compute. Quantization is primarily a technique to speed up inference and only the forward pass is supported for quantized operators."""
    T2:str = """Quantization Aware Training (QAT) models the effects of quantization during training allowing for higher accuracy compared to other quantization methods. We can do QAT for static, dynamic or weight only quantization. During training, all calculations are done in floating point, with fake_quant modules modeling the effects of quantization by clamping and rounding to simulate the effects of INT8."""
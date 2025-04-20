# MechInterp
Mechanistic Interpretibility of PTQ for LLMs Project – a work in progress...

## Transformer
For LLaMA, OLMo architectures.
Specifically tested for:
- NousResearch/DeepHermes-3-Llama-3-3B-Preview, NousResearch/DeepHermes-3-Llama-3-8B-Preview

## Quantization (Post-Hoq)
- PTQ (PTDQ, PTSQ) for transformers using ptq utils.
- FFQ (Fusion FrameQuant) for transformers using ffq utils. 

## Mechanistic Interpretibility Methods
- Topk-1 Logit Lens (nostalgebraist w. mods: https://github.com/nostalgebraist/transformer-utils/tree/main/src/transformer_utils) for single model.
- Topk-1 Comparison Logit Lens for two models, using one model as true distribution.
- Topk-n Logit Lens for single model.

### lm_tasks:
- Chatbot analysis to gather cost and performance metrics
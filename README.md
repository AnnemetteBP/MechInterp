# MechInterp
Mechanistic Interpretibility for quantized LLMs – a work in progress...

## Transformers
For LLaMA, OLMo architectures.
Specifically tested for:
- NousResearch/DeepHermes-3-Llama-3-3B-Preview, NousResearch/DeepHermes-3-Llama-3-8B-Preview
- NousResearch/Llama-3.2-1B, allenai/OLMo-1B-0724-hf

## Quantization (PTQ)
- (1 issues atm.), 1.58, 2, 4 - and 8-bit.
- Uniform symmetric and asymmetric.
- PTDQ and PTSQ and FFQ (Fusion FrameQuant) 

## Mechanistic Interpretibility Methods
- Topk-1 Logit Lens (nostalgebraist w. mods: https://github.com/nostalgebraist/transformer-utils/tree/main/src/transformer_utils) for single model.
- Topk-1 Comparison Logit Lens for two models, using one model as true distribution.
- Topk-n Logit Lens for single model.
- SAE html, heatmap and concept plotting for single - and multi-model comparison.

### chatbot_analysis:
- Chatbot analysis to gather cost and performance metrics by logging to json.
- Plot analysis results in various ways from parallel coordinates plots to subplots.
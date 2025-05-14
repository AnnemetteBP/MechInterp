# MechInterp
Mechanistic Interpretibility for quantized LLMs – a work in progress...

## Transformers
For LLaMA, OLMo architectures.


## Developments
Visualizations use plotly and adds TopK-N predictions to hover.
### The Logit Lens
- The TopK-N Logit Lens of logits, probs, kl, ranks and entropy: topk_lens_plotter.py
- The TopK-N Comparing Logit Lens of js and nwd: topk_comparing_plotter.py

### SAE
- SAE Saliency TopK-N Heatmap: sae_heatmap_plotting.py
- SAE Saliency TopK-N Heatmap: heatmap_comparing_plotting.py

NOTE: This lens and SAE utilities are precision-safe and supports float16, bfloat16, and quantized formats like int8.
However, it does not apply dequantization scaling to int8/uint8 tensors.
Predictions based on raw int values may be structurally meaningful but are not guaranteed to reflect actual model behavior without scale factors.

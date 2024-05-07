# Grounding Eval
Goal: Benchmark methods on the task of grounding (ie. given a natural language description of a webelement, return it's bounding box.)

## How to run

### CogAgent

Run:
```bash
cd CogAgent
python inference_hf.py --from_pretrained THUDM/cogagent-chat-hf
```

To run the quantized version of the model, use:
```bash
cd CogAgent
python inference_hf.py --from_pretrained THUDM/cogagent-chat-hf --quant 4
```

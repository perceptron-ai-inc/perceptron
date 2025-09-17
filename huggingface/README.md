# Isaac on HuggingFace 

## Loading Isaac v0.1
```python
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from huggingface.modular_isaac import IsaacProcessor

tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True, use_fast=False)
config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)
processor = IsaacProcessor(tokenizer=tokenizer, config=config)
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
```

## Example using our `main.py` script

Example input: `Determine whether it is safe to cross the street. Look for signage and moving traffic.` 

![input](huggingface/assets/example.webp)


```bash
$ python huggingface/main.py 

...

Full generated output:
<|endoftext|><think>

</think>

No, it is not safe to cross the street at this time. The <point_box mention="traffic light"> (808,248) (863,386) </point_box> in the background is showing a red signal, which means it's not safe or legal to cross the street. This red light indicates that vehicles have the right of way, and pedestrians should wait until the light changes before proceeding to cross. It's important to always follow traffic signals for your safety and the safety of others on the road. When in doubt, it's best to wait until the signal changes to green before crossing the street.<|im_end|>
```

Visualizing the results 

![prediction](huggingface/assets/prediction.jpeg)

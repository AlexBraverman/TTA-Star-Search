import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMWrapper:
    """Thin wrapper around a HuggingFace causal LM."""

    def __init__(self, model_name: str, dtype=torch.float16):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self):
        return next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Raw prompt generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def chat(self, messages: list[dict], max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Chat-template generation (falls back to raw if unsupported)."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(m["content"] for m in messages)
        return self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)

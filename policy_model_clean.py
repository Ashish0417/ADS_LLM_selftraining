"""
Minimal, clean PolicyModel + ModelManager used by main.py.
This module is a safe, single-responsibility replacement for the broken
`policy_model.py` that had concatenated/duplicate contents.

It supports a light-weight PolicyModel (load, generate, get_model_size) and
ModelManager.initialize_models() and getters used by main.py.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Optional PEFT
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    LoraConfig = None
    get_peft_model = None
    PEFT_AVAILABLE = False

from config import MODEL_CONFIG
from utils import TextProcessor


class PolicyModel:
    def __init__(self, model_name: str = MODEL_CONFIG.policy_model, device: str = MODEL_CONFIG.device):
        self.model_name = model_name
        self.device = device
        self.lora_applied = False

        logger.info(f"Loading policy model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForSeq2SeqLM.from_pretrained(model_name, low_cpu_mem_usage=True, trust_remote_code=True)

        # Try to apply PEFT/LoRA if available
        if PEFT_AVAILABLE and LoraConfig and get_peft_model:
            try:
                lora_cfg = LoraConfig(
                    r=MODEL_CONFIG.lora_r,
                    lora_alpha=MODEL_CONFIG.lora_alpha,
                    target_modules=["q", "k", "v", "o", "wi", "wo"],
                    lora_dropout=MODEL_CONFIG.lora_dropout,
                    bias="none",
                    task_type="SEQ_2_SEQ_LM",
                )
                wrapped = get_peft_model(base, lora_cfg)
                # freeze base, enable adapter params
                for n, p in wrapped.named_parameters():
                    p.requires_grad = False
                adapter_names = [n for n, _ in wrapped.named_parameters() if "lora" in n.lower() or "adapter" in n.lower()]
                if adapter_names:
                    for n, p in wrapped.named_parameters():
                        if "lora" in n.lower() or "adapter" in n.lower():
                            p.requires_grad = True
                    self.model = wrapped
                    self.lora_applied = True
                    logger.info(f"LoRA applied; adapter params: {len(adapter_names)}")
                else:
                    self.model = wrapped
            except Exception as e:
                logger.warning(f"PEFT wrap failed: {e}; using base model")
                self.model = base
        else:
            self.model = base

        self.model.to(self.device)
        self.model.eval()

    def generate(self, instruction: str, context: str = "", max_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9) -> str:
        prompt = f"{context}\n\n{instruction}" if context else instruction
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_tokens, do_sample=True,
                                      temperature=temperature, top_p=top_p, pad_token_id=self.tokenizer.eos_token_id)
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return TextProcessor.clean_text(txt)

    def get_model_size(self) -> Dict[str, Any]:
        tot = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters": int(tot), "trainable_parameters": int(train), "model_name": self.model_name}


class ModelManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.policy_model: PolicyModel = None

    def initialize_models(self):
        model_name = self.config.get('policy_model_name', MODEL_CONFIG.policy_model)
        device = self.config.get('device', MODEL_CONFIG.device)
        self.policy_model = PolicyModel(model_name=model_name, device=device)
        logger.info("PolicyModel initialized")

    def get_policy_model(self) -> PolicyModel:
        return self.policy_model

    def get_optimizer_model(self):
        # not used by main; kept for compatibility
        return None

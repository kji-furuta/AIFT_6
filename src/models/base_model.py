from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype or torch.float16
        
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    @abstractmethod
    def load_model(self) -> PreTrainedModel:
        pass
    
    @abstractmethod
    def load_tokenizer(self) -> PreTrainedTokenizer:
        pass
    
    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, 
                 top_p: float = 0.9, top_k: int = 50, do_sample: bool = True, **kwargs) -> str:
        """Generate text using the loaded model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded. Call load_tokenizer() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        
        # Move inputs to device
        if self.device.type == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Clear cache before generation for large models
        if hasattr(self, '_is_large_model') and self._is_large_model():
            torch.cuda.empty_cache()
        
        # Generate with memory-efficient settings
        try:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs
                )
        except torch.cuda.OutOfMemoryError:
            logger.error("CUDA out of memory during generation. Trying with reduced settings...")
            torch.cuda.empty_cache()
            # Retry with reduced settings
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=min(max_new_tokens, 256),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,  # Disable KV cache
                **kwargs
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text

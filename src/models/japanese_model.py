import torch
from typing import Optional, Dict, Any, List, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
import logging
import gc
from .base_model import BaseModel
from ..utils.gpu_utils import optimize_model_for_gpu, get_gpu_memory_info

logger = logging.getLogger(__name__)


class JapaneseModel(BaseModel):
    """Japanese Language Model for fine-tuning and inference"""
    
    # Supported Japanese models
    SUPPORTED_MODELS = {
        # Ultra-large models (50B+)
        "meta-llama/Llama-3.1-70B-Instruct": {
            "display_name": "Meta Llama 3.1 70B Instruct (Japanese capable)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 140,
            "requires_auth": True,
        },
        "microsoft/WizardLM-2-8x22B": {
            "display_name": "WizardLM-2 8x22B (Japanese capable)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 176,
        },
        "tokyotech-llm/Swallow-70b-instruct-hf": {
            "display_name": "Swallow 70B Instruct Japanese",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 140,
        },
        
        # Large models (30B-50B)
        "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese": {
            "display_name": "CyberAgent DeepSeek-R1 Distill Qwen 32B Japanese",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 64,
            "memory_optimized": True,
            "max_memory_per_gpu": "22GB",
        },
        "meta-llama/Llama-3.1-8B-Instruct": {
            "display_name": "Meta Llama 3.1 8B Instruct (Japanese capable)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 16,
            "requires_auth": True,
        },
        "microsoft/Phi-3.5-mini-instruct": {
            "display_name": "Microsoft Phi-3.5 Mini Instruct (3.8B)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 8,
        },
        "Qwen/Qwen2.5-32B-Instruct": {
            "display_name": "Qwen 2.5 32B Instruct (Japanese capable)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 64,
        },
        "microsoft/Orca-2-13b": {
            "display_name": "Microsoft Orca-2 13B",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 26,
        },
        
        # Large-medium models (14B-30B)
        "Qwen/Qwen2.5-14B-Instruct": {
            "display_name": "Qwen 2.5 14B Instruct (Japanese capable)",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 28,
        },
        "tokyotech-llm/Swallow-13b-instruct-hf": {
            "display_name": "Swallow 13B Instruct Japanese",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 26,
        },
        "elyza/ELYZA-japanese-Llama-2-13b-instruct": {
            "display_name": "ELYZA Japanese Llama-2 13B Instruct",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 26,
        },
        
        # Medium-large models (10B-17B)
        "stabilityai/japanese-stablelm-instruct-alpha-7b-v2": {
            "display_name": "Japanese StableLM Alpha 7B v2 Instruct",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 14,
        },
        "rinna/youri-7b-chat": {
            "display_name": "Rinna Youri 7B Chat",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 14,
        },
        "cyberagent/open-calm-7b": {
            "display_name": "CyberAgent OpenCALM 7B",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 14,
        },
        "matsuo-lab/weblab-10b-instruction-sft": {
            "display_name": "Matsuo Lab WebLab 10B Instruction",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 20,
        },
        
        # Medium models (8B-10B)
        "elyza/Llama-3-ELYZA-JP-8B": {
            "display_name": "Llama-3 ELYZA Japanese 8B",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 16,
        },
        
        # Small-medium models (3B-7B)
        "stabilityai/japanese-stablelm-3b-4e1t-instruct": {
            "display_name": "Japanese StableLM 3B Instruct",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 8,
        },
        "rinna/japanese-gpt-neox-3.6b-instruction-sft": {
            "display_name": "Rinna GPT-NeoX 3.6B Instruct",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 8,
        },
        "line-corporation/japanese-large-lm-3.6b-instruction-sft": {
            "display_name": "LINE Japanese LM 3.6B Instruct",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 8,
        },
        
        # Compact models (1B-3B)
        "rinna/japanese-gpt-1b": {
            "display_name": "Rinna GPT 1B Japanese",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 4,
        },
        "cyberagent/open-calm-1b": {
            "display_name": "CyberAgent OpenCALM 1B",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 4,
        },
        "cyberagent/open-calm-3b": {
            "display_name": "CyberAgent OpenCALM 3B",
            "model_type": "causal_lm",
            "recommended_dtype": torch.float16,
            "min_gpu_memory_gb": 8,
        }
    }
    
    def __init__(
        self,
        model_name: str = "stabilityai/japanese-stablelm-3b-4e1t-instruct",
        device: Optional[torch.device] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        use_flash_attention: bool = True,
        gradient_checkpointing: bool = False,
        use_auth_token: Optional[str] = None,
        use_qlora: bool = False,
        qlora_r: int = 64,
        qlora_alpha: int = 16,
        qlora_dropout: float = 0.1,
        enable_deepspeed: bool = False,
        deepspeed_config: Optional[Dict] = None,
        offload_optimizer: bool = False,
        offload_param: bool = False,
        cpu_offload: bool = False,
        disk_offload_dir: Optional[str] = None
    ):
        # Check model support
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} is not in the supported list. Proceeding anyway.")
        
        # Check if model requires authentication
        if (model_name in self.SUPPORTED_MODELS and 
            self.SUPPORTED_MODELS[model_name].get("requires_auth", False) and 
            not use_auth_token):
            logger.warning(
                f"Model {model_name} requires HuggingFace authentication. "
                "Please provide use_auth_token parameter."
            )
        
        super().__init__(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            torch_dtype=torch_dtype
        )
        
        self.use_flash_attention = use_flash_attention
        self.gradient_checkpointing = gradient_checkpointing
        self.use_auth_token = use_auth_token
        self.use_qlora = use_qlora
        self.qlora_r = qlora_r
        self.qlora_alpha = qlora_alpha
        self.qlora_dropout = qlora_dropout
        self.enable_deepspeed = enable_deepspeed
        self.deepspeed_config = deepspeed_config
        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param
        self.cpu_offload = cpu_offload
        self.disk_offload_dir = disk_offload_dir
        
        # Auto-enable optimizations for large models
        if self._is_large_model():
            if not gradient_checkpointing:
                logger.info("Auto-enabling gradient checkpointing for large model")
                self.gradient_checkpointing = True
            if not load_in_8bit and not load_in_4bit:
                if self._get_model_size_category() in ["ultra_large", "large"]:
                    logger.info("Auto-enabling 4-bit quantization for ultra-large model")
                    self.load_in_4bit = True
                    self.use_qlora = True
                else:
                    logger.info("Auto-enabling 8-bit quantization for large model")
                    self.load_in_8bit = True
        
        # Ensure QLoRA is enabled if any quantization is used for fine-tuning
        if (self.load_in_8bit or self.load_in_4bit) and not self.use_qlora:
            logger.warning("Quantization (8-bit or 4-bit) is enabled, but use_qlora is False. Setting use_qlora to True for fine-tuning.")
            self.use_qlora = True
    
    def _is_large_model(self) -> bool:
        """Check if the model is considered large (17B+)"""
        if self.model_name in self.SUPPORTED_MODELS:
            required_memory = self.SUPPORTED_MODELS[self.model_name]["min_gpu_memory_gb"]
            return required_memory >= 34  # 17B+ models
        return False
    
    def _get_model_size_category(self) -> str:
        """Get model size category"""
        if self.model_name in self.SUPPORTED_MODELS:
            memory_gb = self.SUPPORTED_MODELS[self.model_name]["min_gpu_memory_gb"]
            if memory_gb >= 100:
                return "ultra_large"
            elif memory_gb >= 50:
                return "large"
            elif memory_gb >= 30:
                return "large_medium"
            elif memory_gb >= 15:
                return "medium_large"
            elif memory_gb >= 10:
                return "medium"
            elif memory_gb >= 5:
                return "small_medium"
            else:
                return "compact"
        return "unknown"
    
    def get_recommended_training_config(self) -> Dict[str, Any]:
        """Get recommended training configuration for the model"""
        category = self._get_model_size_category()
        
        if category == "ultra_large":
            return {
                "learning_rate": 1e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 32,
                "num_epochs": 2,
                "gradient_checkpointing": True,
                "fp16": True,
                "use_qlora": True,
                "qlora_4bit": True,
                "qlora_r": 128,
                "qlora_alpha": 32,
                "model_parallel": True,
                "deepspeed": True,
                "deepspeed_stage": 3,
                "offload_optimizer": True,
                "offload_param": True,
                "cpu_offload": True,
                "max_seq_length": 512,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100
            }
        elif category == "large":
            return {
                "learning_rate": 2e-4,
                "batch_size": 1,
                "gradient_accumulation_steps": 16,
                "num_epochs": 3,
                "gradient_checkpointing": True,
                "fp16": True,
                "use_qlora": True,
                "qlora_4bit": True,
                "qlora_r": 64,
                "qlora_alpha": 16,
                "model_parallel": True,
                "deepspeed": True,
                "deepspeed_stage": 2,
                "offload_optimizer": True,
                "max_seq_length": 1024,
                "logging_steps": 10,
                "save_steps": 200,
                "eval_steps": 200
            }
        elif category == "large_medium":
            return {
                "learning_rate": 2e-4,
                "batch_size": 2,
                "gradient_accumulation_steps": 8,
                "num_epochs": 3,
                "gradient_checkpointing": True,
                "fp16": True,
                "use_qlora": True,
                "qlora_4bit": False,
                "qlora_r": 32,
                "qlora_alpha": 8,
                "deepspeed": True,
                "deepspeed_stage": 2,
                "max_seq_length": 2048,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500
            }
        elif category in ["medium_large", "medium"]:
            return {
                "learning_rate": 3e-4,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 4,
                "gradient_checkpointing": True,
                "fp16": True,
                "use_qlora": True,
                "qlora_4bit": False,
                "qlora_r": 16,
                "qlora_alpha": 4,
                "max_seq_length": 2048,
                "logging_steps": 10,
                "save_steps": 1000,
                "eval_steps": 1000
            }
        else:
            return {
                "learning_rate": 3e-4,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "num_epochs": 5,
                "gradient_checkpointing": False,
                "fp16": True,
                "use_qlora": False,
                "max_seq_length": 2048,
                "logging_steps": 10,
                "save_steps": 1000,
                "eval_steps": 1000
            }
    
    def check_gpu_requirements(self) -> bool:
        """Check if GPU requirements are met"""
        gpu_info = get_gpu_memory_info()
        
        if not gpu_info["available"]:
            logger.warning("No GPU available. Model loading may be slow or fail.")
            return False
        
        if self.model_name in self.SUPPORTED_MODELS:
            required_memory = self.SUPPORTED_MODELS[self.model_name]["min_gpu_memory_gb"]
            
            for device_info in gpu_info["devices"]:
                if device_info["free_memory_gb"] >= required_memory:
                    logger.info(
                        f"GPU {device_info['name']} has sufficient memory "
                        f"({device_info['free_memory_gb']:.1f}GB free, "
                        f"{required_memory}GB required)"
                    )
                    return True
            
            logger.warning(
                f"Insufficient GPU memory for {self.model_name}. "
                f"Required: {required_memory}GB, "
                f"Available: {max(d['free_memory_gb'] for d in gpu_info['devices']):.1f}GB. "
                f"Consider using quantization (8-bit or 4-bit)."
            )
        
        return True  # Allow loading anyway
    
    def get_advanced_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get advanced quantization configuration for large models"""
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True
            )
        return None
    
    def get_deepspeed_config(self) -> Dict[str, Any]:
        """Get DeepSpeed configuration for large model training"""
        if not self.enable_deepspeed:
            return {}
        
        category = self._get_model_size_category()
        
        if category == "ultra_large":
            return {
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "sub_group_size": 1e9,
                    "reduce_bucket_size": "auto",
                    "stage3_prefetch_bucket_size": "auto",
                    "stage3_param_persistence_threshold": "auto",
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_gather_16bit_weights_on_model_save": True
                },
                "fp16": {
                    "enabled": True,
                    "auto_cast": False,
                    "loss_scale": 0,
                    "initial_scale_power": 32,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "steps_per_print": 10,
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto",
                "wall_clock_breakdown": False
            }
        elif category == "large":
            return {
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True
                    },
                    "allgather_partitions": True,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": True
                },
                "fp16": {
                    "enabled": True
                },
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": "auto",
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto"
            }
        else:
            return {
                "zero_optimization": {
                    "stage": 1
                },
                "fp16": {
                    "enabled": True
                },
                "gradient_accumulation_steps": "auto",
                "train_batch_size": "auto",
                "train_micro_batch_size_per_gpu": "auto"
            }
    
    def load_model(self) -> PreTrainedModel:
        """Load the model with advanced optimizations"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Check GPU requirements
        self.check_gpu_requirements()
        
        # Clear cache before loading large models
        if self._is_large_model():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Get quantization config
        quantization_config = self.get_advanced_quantization_config()
        
        # Model loading arguments
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        
        # Advanced device mapping for large models
        if self.device.type == "cuda":
            category = self._get_model_size_category()
            if category in ["ultra_large", "large"]:
                # Custom device map for optimal memory distribution
                model_kwargs["device_map"] = "balanced"
                
                # Check for memory-optimized models
                if self.model_name in self.SUPPORTED_MODELS:
                    model_config = self.SUPPORTED_MODELS[self.model_name]
                    if model_config.get("memory_optimized", False):
                        max_mem = model_config.get("max_memory_per_gpu", "22GB")
                        model_kwargs["max_memory"] = {i: max_mem for i in range(torch.cuda.device_count())}
                    else:
                        model_kwargs["max_memory"] = {i: "80GB" for i in range(torch.cuda.device_count())}
                else:
                    model_kwargs["max_memory"] = {i: "80GB" for i in range(torch.cuda.device_count())}
                    
                model_kwargs["offload_folder"] = self.disk_offload_dir or "offload"
                model_kwargs["offload_state_dict"] = True
            else:
                model_kwargs["device_map"] = "auto"
        
        # Add authentication token if required
        if self.use_auth_token:
            model_kwargs["token"] = self.use_auth_token
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        # Flash Attention setup
        if self.use_flash_attention and self.device.type == "cuda":
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            except:
                logger.warning("Flash Attention 2 not available, using default attention")
        
        # Large model optimizations
        if self._is_large_model():
            model_kwargs["low_cpu_mem_usage"] = True
            if not self.load_in_8bit and not self.load_in_4bit:
                model_kwargs["torch_dtype"] = torch.float16
                
            # Additional optimizations for DeepSeek model
            if "DeepSeek-R1-Distill-Qwen-32B" in self.model_name:
                # Force 4-bit quantization for this model
                if not self.load_in_4bit:
                    logger.info("Enabling 4-bit quantization for DeepSeek-R1-Distill-Qwen-32B to reduce memory usage")
                    self.load_in_4bit = True
                    quantization_config = self.get_advanced_quantization_config()
                    model_kwargs["quantization_config"] = quantization_config
        
        try:
            # Load model with retry logic for large models
            max_retries = 3 if self._is_large_model() else 1
            for attempt in range(max_retries):
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
                    break
                except torch.cuda.OutOfMemoryError:
                    if attempt < max_retries - 1:
                        logger.warning(f"OOM error, retrying with more aggressive memory optimizations (attempt {attempt + 1}/{max_retries})")
                        torch.cuda.empty_cache()
                        gc.collect()
                        # Enable more aggressive quantization
                        if not self.load_in_4bit:
                            self.load_in_4bit = True
                            quantization_config = self.get_advanced_quantization_config()
                            model_kwargs["quantization_config"] = quantization_config
                    else:
                        raise
            
            # Apply QLoRA if requested
            if self.use_qlora and (self.load_in_4bit or self.load_in_8bit):
                self._apply_qlora()
            
            # GPU optimization
            if self.device.type == "cuda":
                self.model, self.device = optimize_model_for_gpu(
                    self.model,
                    self.device,
                    enable_mixed_precision=not (self.load_in_8bit or self.load_in_4bit),
                    gradient_checkpointing=self.gradient_checkpointing
                )
            
            # Enable gradient checkpointing for training
            if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Log memory usage
            if self.device.type == "cuda":
                allocated_memory = torch.cuda.memory_allocated() / 1024**3
                reserved_memory = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"GPU memory: {allocated_memory:.2f}GB allocated, {reserved_memory:.2f}GB reserved")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        return self.model
    
    def _apply_qlora(self):
        """Apply QLoRA configuration to the model"""
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            # Prepare model for k-bit training
            self.model = prepare_model_for_kbit_training(self.model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                r=self.qlora_r,
                lora_alpha=self.qlora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.qlora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # Apply LoRA
            self.model = get_peft_model(self.model, lora_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"QLoRA applied - trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%), "
                f"all params: {total_params:,}"
            )
            
        except ImportError:
            logger.warning("PEFT not installed, skipping QLoRA")
        except Exception as e:
            logger.warning(f"Failed to apply QLoRA: {e}")
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer"""
        logger.info(f"Loading tokenizer for: {self.model_name}")
        
        try:
            tokenizer_kwargs = {
                "pretrained_model_name_or_path": self.model_name,
                "trust_remote_code": True
            }
            
            # Add authentication token if required
            if self.use_auth_token:
                tokenizer_kwargs["token"] = self.use_auth_token
            
            self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_kwargs)
            
            # Set pad token if not available
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        return self.tokenizer
    
    def format_prompt(self, instruction: str, input_text: Optional[str] = None) -> str:
        """Format prompt for Japanese models"""
        # Model-specific prompt formatting
        if "stablelm" in self.model_name.lower():
            if input_text:
                prompt = f"<|system|>あなたは日本語でサポートするAIアシスタントです。<|endoftext|>\n<|user|>{instruction}\n入力: {input_text}<|endoftext|>\n<|assistant|>"
            else:
                prompt = f"<|system|>あなたは日本語でサポートするAIアシスタントです。<|endoftext|>\n<|user|>{instruction}<|endoftext|>\n<|assistant|>"
        elif "llama" in self.model_name.lower():
            if input_text:
                prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 回答:"
            else:
                prompt = f"### 指示:\n{instruction}\n\n### 回答:"
        elif "rinna" in self.model_name.lower():
            if input_text:
                prompt = f"指示: {instruction}\n入力: {input_text}\n回答: "
            else:
                prompt = f"指示: {instruction}\n回答: "
        else:
            # Default format for other models
            if input_text:
                prompt = f"指示: {instruction}\n入力: {input_text}\n回答: "
            else:
                prompt = f"指示: {instruction}\n回答: "
        
        return prompt
    
    def generate_japanese(
        self,
        instruction: str,
        input_text: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate Japanese text response"""
        prompt = self.format_prompt(instruction, input_text)
        return self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            **kwargs
        )
    
    def load_with_fallback(self, fallback_models: Optional[List[str]] = None) -> bool:
        """Load model with fallback options"""
        if fallback_models is None:
            # Default fallback models in order of preference
            fallback_models = [
                "stabilityai/japanese-stablelm-3b-4e1t-instruct",
                "rinna/japanese-gpt-neox-3.6b-instruction-sft",
                "line-corporation/japanese-large-lm-3.6b-instruction-sft",
                "elyza/Llama-3-ELYZA-JP-8B",
            ]
        
        # Try primary model first
        try:
            self.load_model()
            self.load_tokenizer()
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
        
        # Try fallback models
        for fallback_model in fallback_models:
            if fallback_model == self.model_name:
                continue
            
            logger.info(f"Trying fallback model: {fallback_model}")
            self.model_name = fallback_model
            
            try:
                self.load_model()
                self.load_tokenizer()
                logger.info(f"Successfully loaded fallback model: {fallback_model}")
                return True
            except Exception as e:
                logger.error(f"Failed to load {fallback_model}: {e}")
        
        logger.error("All models failed to load")
        return False
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, Any]:
        """List all supported models"""
        return cls.SUPPORTED_MODELS
    
    @classmethod
    def list_large_models(cls) -> Dict[str, Any]:
        """List only large models (17B+)"""
        large_models = {}
        for model_name, config in cls.SUPPORTED_MODELS.items():
            if config.get("min_gpu_memory_gb", 0) >= 34:  # 17B+ models
                large_models[model_name] = config
        return large_models
    
    @classmethod
    def list_models_by_size(cls) -> Dict[str, List[str]]:
        """List models grouped by size"""
        models_by_size = {
            "ultra_large": [],  # 50B+
            "large": [],        # 30B-50B
            "large_medium": [], # 17B-30B
            "medium_large": [], # 10B-17B
            "medium": [],       # 8B-10B
            "small_medium": [], # 3B-7B
            "compact": []       # 1B-3B
        }
        
        for model_name, config in cls.SUPPORTED_MODELS.items():
            memory_gb = config.get("min_gpu_memory_gb", 0)
            if memory_gb >= 100:
                models_by_size["ultra_large"].append(model_name)
            elif memory_gb >= 50:
                models_by_size["large"].append(model_name)
            elif memory_gb >= 30:
                models_by_size["large_medium"].append(model_name)
            elif memory_gb >= 15:
                models_by_size["medium_large"].append(model_name)
            elif memory_gb >= 10:
                models_by_size["medium"].append(model_name)
            elif memory_gb >= 5:
                models_by_size["small_medium"].append(model_name)
            else:
                models_by_size["compact"].append(model_name)
        
        return models_by_size
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        if model_name not in cls.SUPPORTED_MODELS:
            return {"error": f"Model {model_name} not found in supported models"}
        
        config = cls.SUPPORTED_MODELS[model_name]
        return {
            "model_name": model_name,
            "display_name": config["display_name"],
            "model_type": config["model_type"],
            "recommended_dtype": str(config["recommended_dtype"]),
            "min_gpu_memory_gb": config["min_gpu_memory_gb"],
            "requires_auth": config.get("requires_auth", False),
            "is_large_model": config.get("min_gpu_memory_gb", 0) >= 34,
            "recommended_training_config": cls().get_recommended_training_config() if model_name == cls().model_name else None
        } 
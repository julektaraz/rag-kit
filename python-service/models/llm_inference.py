"""
LLM Inference Module

Loads and runs language models with 4-bit quantization for GPU efficiency.
Uses transformers and bitsandbytes for quantized inference.
"""

import logging
from typing import List, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# Import default constants
try:
    from config import (
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_TOP_P,
        DEFAULT_DO_SAMPLE,
    )
except ImportError:
    # Fallback defaults
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_P = 0.9
    DEFAULT_DO_SAMPLE = True

logger = logging.getLogger(__name__)


class LLMModel:
    """
    Manages LLM model loading and text generation.
    
    Supports 4-bit quantization (NF4) to fit larger models in GPU memory.
    """
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_quantization: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Hugging Face model identifier
            use_quantization: Whether to use 4-bit quantization (NF4)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.use_quantization = use_quantization and device == "cuda"
        
        logger.info(f"Loading LLM: {model_name} on {device}")
        logger.info(f"Quantization: {self.use_quantization}")
        
        # Configure quantization if using GPU
        quantization_config = None
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using 4-bit NF4 quantization")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if quantization_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
            )
            if device != "cuda":
                self.model = self.model.to(device)
        
        # Create text generation pipeline
        # When using device_map="auto" with quantization, don't pass device to pipeline
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
        }
        # Only add device if not using quantization (which uses device_map="auto")
        if not self.use_quantization:
            pipeline_kwargs["device"] = self.device
        
        self.pipeline = pipeline("text-generation", **pipeline_kwargs)
        
        logger.info("LLM model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        do_sample: bool = DEFAULT_DO_SAMPLE,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            
        Returns:
            Generated text
        """
        # Generate response
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            return_full_text=False,  # Don't repeat the prompt
            pad_token_id=self.tokenizer.eos_token_id,
        )
        
        # Extract generated text
        generated_text = outputs[0]["generated_text"]
        
        return generated_text.strip()
    
    def format_prompt_with_context(
        self,
        query: str,
        contexts: List[str],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Format a RAG prompt with retrieved context.
        
        Args:
            query: User's question
            contexts: List of retrieved context chunks
            system_prompt: Optional system instruction
            
        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. Answer the question based on "
                "the provided context. If the answer is not in the context, "
                "say so clearly."
            )
        
        # Format context
        context_text = "\n\n".join(
            [f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)]
        )
        
        # Format full prompt based on model type
        if "mistral" in self.model_name.lower() or "instruct" in self.model_name.lower():
            # Use instruction format
            prompt = f"""<s>[INST] {system_prompt}

Context:
{context_text}

Question: {query}

Answer: [/INST]"""
        elif "gemma" in self.model_name.lower():
            # Gemma format
            prompt = f"""<start_of_turn>user
{system_prompt}

Context:
{context_text}

Question: {query}<end_of_turn>
<start_of_turn>model
"""
        else:
            # Generic format
            prompt = f"""{system_prompt}

Context:
{context_text}

Question: {query}

Answer:"""
        
        return prompt


__all__ = ["LLMModel"]


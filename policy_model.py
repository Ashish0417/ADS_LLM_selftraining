import logging
import torch
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
from utils import TextProcessor

logger = logging.getLogger(__name__)


class PolicyModel:
    """Wrapper for policy model - T5 Encoder-Decoder"""
    
    def __init__(self, model_name: str, device: str = "cpu", load_in_8bit: bool = False):
        """Initialize policy model"""
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        logger.info(f"Loading policy model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - use Seq2SeqLM for T5 (encoder-decoder)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        self.model.eval()
        logger.info(f"Policy model loaded successfully")
    
    def generate(self, instruction: str, context: str = "", max_tokens: int = 256,
                temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response"""
        
        # Build prompt
        if context:
            prompt = f"{context}\n\n{instruction}"
        else:
            prompt = instruction
        
        try:
            # Tokenize with attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding='max_length'
            )
            
            input_ids = inputs['input_ids']
            attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
            
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Generate
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_tokens + input_ids.shape[1],
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            return TextProcessor.clean_text(response) if response else "No response"
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def generate_batch(self, instructions: List[str], batch_size: int = 1) -> List[str]:
        """Generate responses for batch of instructions"""
        responses = []
        
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i:i+batch_size]
            for instruction in batch:
                response = self.generate(instruction)
                responses.append(response)
        
        return responses
    
    def in_context_learn(self, instruction: str, examples: List[Dict], 
                        max_tokens: int = 256) -> str:
        """Generate with in-context examples"""
        
        # Build prompt with examples
        prompt = "You are a helpful assistant.\n\n"
        
        for example in examples[:3]:
            prompt += f"Input: {example.get('input', example.get('instruction', ''))}\n"
            prompt += f"Output: {example.get('output', example.get('response', ''))}\n\n"
        
        prompt += f"Input: {instruction}\nOutput:"
        
        return self.generate(prompt, max_tokens=max_tokens)
    
    def evaluate_response(self, instruction: str, response: str) -> float:
        """Simple heuristic evaluation (0-1)"""
        from utils import HeuristicScorer
        return HeuristicScorer.score_response(instruction, response)
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_name': self.model_name,
        }


class OptimizerModel:
    """Wrapper for optimizer model"""
    
    def __init__(self, policy_model: PolicyModel):
        """Initialize optimizer model"""
        self.model = policy_model
        self.generation_history = []
        logger.info("Optimizer model initialized from policy model")
    
    def analyze_task(self, observed_instructions: List[str]) -> str:
        """Analyze task requirements"""
        
        if not observed_instructions:
            return ""
        
        analysis_prompt = f"""Analyze the following instructions and identify:
1. The common themes or requirements
2. Knowledge gaps that need to be filled
3. Capabilities that need improvement

Instructions:
{chr(10).join(f'{i+1}. {instr}' for i, instr in enumerate(observed_instructions[:3]))}

Analysis:"""
        
        return self.model.generate(analysis_prompt, max_tokens=300)
    
    def generate_api_trajectory(self, observed_instructions: List[str], 
                               available_apis: List[str] = None) -> List[Dict]:
        """Generate API trajectory"""
        
        if available_apis is None:
            available_apis = ['information_retrieval', 'demonstration_generation', 
                            'question_answering']
        
        if not observed_instructions:
            return [{'name': 'none', 'param': ''}]
        
        # Build prompt
        api_prompt = f"""Based on the following instructions, determine which APIs to call.

Available APIs:
1. information_retrieval(query): Retrieve relevant documents
2. demonstration_generation(topic): Generate examples
3. question_answering(question): Answer complex questions

Instructions:
{chr(10).join(f'{i+1}. {instr}' for i, instr in enumerate(observed_instructions[:3]))}

Determine the best APIs to call:
<api calls>
<api>information_retrieval(relevant query)</api>
</api calls>

Trajectory:"""
        
        response = self.model.generate(api_prompt, max_tokens=400)
        self.generation_history.append({
            'instructions': observed_instructions,
            'trajectory': response
        })
        
        from utils import APITrajectoryParser
        return APITrajectoryParser.parse_trajectory(response)
    
    def refine_with_trajectory(self, trajectory_response: str, reward: float):
        """Update optimizer with trajectory feedback"""
        logger.info(f"Refining optimizer with reward: {reward}")


class ModelManager:
    """Manage policy and optimizer models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.policy_model = None
        self.optimizer_model = None
    
    def initialize_models(self):
        """Initialize both models"""
        logger.info("=" * 80)
        logger.info("INITIALIZING MODELS")
        logger.info("=" * 80)
        
        # Get model name from config
        model_name = self.config.get('policy_model_name', 'google/flan-t5-base')
        device = self.config.get('device', 'cpu')
        load_in_8bit = self.config.get('load_in_8bit', False)
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"8-bit quantization: {load_in_8bit}")
        
        # Initialize policy model
        self.policy_model = PolicyModel(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit
        )
        
        # Initialize optimizer from policy
        self.optimizer_model = OptimizerModel(self.policy_model)
        
        # Print model info
        model_info = self.policy_model.get_model_size()
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Total Parameters: {model_info['total_parameters']:,}")
        logger.info("=" * 80)
    
    def get_policy_model(self) -> PolicyModel:
        """Get policy model"""
        return self.policy_model
    
    def get_optimizer_model(self) -> OptimizerModel:
        """Get optimizer model"""
        return self.optimizer_model

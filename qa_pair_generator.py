
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)


class QAPairGenerator:
    """
    Generate Question-Answer pairs from retrieved Wikipedia data
    
    COMPATIBLE: Works with existing PolicyModel
    - Uses generate() method
    - Uses in_context_learn() method
    - Does NOT require in_context_learn_improved()
    """
    
    def __init__(self, policy_model=None, device: str = "cpu"):
        """
        Initialize QA pair generator
        
        Args:
            policy_model: Existing policy model (with generate() method)
            device: "cpu" or "cuda"
        """
        self.policy_model = policy_model
        self.device = device
        
        if policy_model is None:
            logger.warning("[QAPairGenerator] No policy model provided, QA generation may be limited")
        
        logger.info("[QAPairGenerator] ✓ Initialized (Compatible Mode)")
    
    def generate_qa_pairs(
        self,
        instruction: str,
        retrieved_data: str,
        num_demonstrations: int = 3,
        num_questions: int = 2
    ) -> Dict[str, Any]:
        """
        Generate Q-A pairs from instruction and retrieved data
        
        COMPATIBLE: Uses existing policy model methods
        
        Args:
            instruction: Original task instruction
            retrieved_data: Retrieved Wikipedia context
            num_demonstrations: Number of demonstration pairs to generate
            num_questions: Number of detailed Q-A pairs to generate
        
        Returns:
            Dictionary with generated Q-A pairs
        """
        
        try:
            result = {
                'instruction': instruction,
                'retrieved_data': retrieved_data[:500],  # Store first 500 chars
                'demonstrations': [],
                'qa_pairs': [],
                'num_demonstrations': num_demonstrations,
                'num_qa_pairs': num_questions
            }
            
            # Step 1: Generate Demonstrations
            logger.info(f"[QAPairGenerator] Generating {num_demonstrations} demonstrations...")
            demonstrations = self._generate_demonstrations(
                instruction,
                retrieved_data,
                num_demonstrations
            )
            result['demonstrations'] = demonstrations
            
            # Step 2: Generate Detailed Q-A Pairs
            logger.info(f"[QAPairGenerator] Generating {num_questions} detailed Q-A pairs...")
            qa_pairs = self._generate_qa_pairs(
                instruction,
                retrieved_data,
                num_questions
            )
            result['qa_pairs'] = qa_pairs
            
            logger.info(f"[QAPairGenerator] ✓ Generated {len(demonstrations)} demonstrations + {len(qa_pairs)} Q-A pairs")
            
            return result
        
        except Exception as e:
            logger.error(f"[QAPairGenerator] Error generating Q-A pairs: {e}")
            return {
                'instruction': instruction,
                'retrieved_data': retrieved_data[:500],
                'demonstrations': [],
                'qa_pairs': [],
                'error': str(e)
            }
    
    def _generate_demonstrations(
        self,
        instruction: str,
        retrieved_data: str,
        num_demonstrations: int
    ) -> List[Dict[str, str]]:
        """
        Generate demonstration (instruction-response) pairs
        Uses EXISTING policy model's generate() method
        """
        demonstrations = []
        
        try:
            # Extract topic from instruction
            topic = self._extract_topic(instruction)
            
            for i in range(num_demonstrations):
                try:
                    # Build prompt for demonstration generation
                    prompt = f"""Based on this topic and context, generate an instruction-response pair:

Topic: {topic}
Context: {retrieved_data[:300]}

Generate a new Instruction and Response pair related to the topic.
Format your answer as:
Instruction: [Your instruction here]
Response: [Your response here]"""
                    
                    # Generate using policy model's generate() method
                    if self.policy_model:
                        response_text = self.policy_model.generate(
                            instruction=prompt,
                            max_tokens=200
                        )
                    else:
                        # Fallback: simple heuristic
                        response_text = f"Instruction: {topic}\nResponse: Based on {topic}, you should..."
                    
                    # Parse response
                    demonstration = self._parse_demonstration(response_text)
                    if demonstration:
                        demonstrations.append(demonstration)
                        logger.debug(f"[QAPairGenerator] Demo {i+1}: {demonstration['instruction'][:60]}...")
                except Exception as e:
                    logger.debug(f"[QAPairGenerator] Error generating demo {i+1}: {e}")
                    continue
            
            return demonstrations
        
        except Exception as e:
            logger.error(f"[QAPairGenerator] Error in demonstration generation: {e}")
            return []
    
    def _generate_qa_pairs(
        self,
        instruction: str,
        retrieved_data: str,
        num_questions: int
    ) -> List[Dict[str, str]]:
        """
        Generate detailed Question-Answer pairs
        Uses EXISTING policy model's in_context_learn() method
        """
        qa_pairs = []
        
        try:
            # Generate specific questions based on context
            questions = self._generate_questions(instruction, retrieved_data, num_questions)
            
            for i, question in enumerate(questions):
                try:
                    # Build examples for in_context_learn
                    examples = [
                        {
                            'input': instruction,
                            'output': retrieved_data[:200]
                        }
                    ]
                    
                    # Generate answer using in_context_learn
                    if self.policy_model:
                        answer_text = self.policy_model.in_context_learn(
                            instruction=question,
                            examples=examples,
                            max_tokens=256
                        )
                    else:
                        # Fallback
                        answer_text = f"Answer to '{question}' based on context..."
                    
                    qa_pair = {
                        'question': question,
                        'answer': answer_text.strip() if answer_text else "No answer",
                        'source': 'generated'
                    }
                    qa_pairs.append(qa_pair)
                    logger.debug(f"[QAPairGenerator] Q-A {i+1}: {question[:60]}...")
                except Exception as e:
                    logger.debug(f"[QAPairGenerator] Error generating Q-A {i+1}: {e}")
                    continue
            
            return qa_pairs
        
        except Exception as e:
            logger.error(f"[QAPairGenerator] Error in Q-A pair generation: {e}")
            return []
    
    def _extract_topic(self, instruction: str) -> str:
        """Extract topic from instruction"""
        # Simple heuristic: use instruction as topic if short, otherwise summarize
        if len(instruction.split()) <= 5:
            return instruction
        
        # For longer instructions, use first few words
        words = instruction.split()[:5]
        return ' '.join(words)
    
    def _generate_questions(
        self,
        instruction: str,
        context: str,
        num_questions: int
    ) -> List[str]:
        """Generate specific questions based on context"""
        questions = []
        
        try:
            prompt = f"""Generate {num_questions} specific, detailed questions about this context:

Context: {context[:300]}

Generate {num_questions} questions (one per line) that test understanding:"""
            
            if self.policy_model:
                response = self.policy_model.generate(
                    instruction=prompt,
                    max_tokens=256
                )
            else:
                # Fallback questions
                response = f"1. What is {instruction}?\n2. How does {instruction} work?"
            
            # Parse questions
            for line in response.split('\n'):
                line = line.strip()
                # Remove numbering (1., 2., etc.)
                if line and len(line) > 5:
                    if line[0].isdigit():
                        line = line.split('.', 1)[1].strip() if '.' in line else line
                    if line and '?' in line:
                        questions.append(line)
            
            return questions[:num_questions]
        
        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            # Return fallback questions
            return [
                f"What is the main topic of this content?",
                f"Can you explain {instruction}?"
            ][:num_questions]
    
    def _parse_demonstration(self, text: str) -> Dict[str, str]:
        """Parse demonstration text into instruction-response"""
        try:
            if 'Instruction:' in text and 'Response:' in text:
                parts = text.split('Response:')
                instruction = parts[0].split('Instruction:')[1].strip()
                response = parts[1].strip()
                return {
                    'instruction': instruction,
                    'response': response,
                    'type': 'demonstration'
                }
            return None
        except Exception as e:
            logger.debug(f"Error parsing demonstration: {e}")
            return None


class QAPairStorage:
    """Store and manage Q-A pairs for each task"""
    
    def __init__(self, output_dir: str = "results/qa_pairs"):
        """Initialize storage"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_qa_data = {}
        
        logger.info(f"[QAPairStorage] Output directory: {self.output_dir}")
    
    def save_qa_pairs(
        self,
        task_number: int,
        task_name: str,
        qa_data: Dict[str, Any]
    ) -> str:
        """
        Save Q-A pairs for a single task
        
        Args:
            task_number: Task number/ID
            task_name: Task name
            qa_data: Generated Q-A data
        
        Returns:
            Path to saved file
        """
        try:
            # Create task-specific entry
            task_entry = {
                'task_number': task_number,
                'task_name': task_name,
                'generated_at': str(__import__('datetime').datetime.now()),
                **qa_data
            }
            
            # Store in memory
            self.all_qa_data[task_number] = task_entry
            
            # Save individual task file
            task_file = self.output_dir / f"task_{task_number:04d}_qa_pairs.json"
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_entry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[QAPairStorage] ✓ Saved to {task_file}")
            
            return str(task_file)
        
        except Exception as e:
            logger.error(f"[QAPairStorage] Error saving Q-A pairs: {e}")
            return ""
    
    def save_all_qa_pairs(self, filename: str = "all_qa_pairs.json") -> str:
        """
        Save all Q-A pairs in single file
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        try:
            output_file = self.output_dir / filename
            
            combined_data = {
                'total_tasks': len(self.all_qa_data),
                'tasks': list(self.all_qa_data.values()),
                'generated_at': str(__import__('datetime').datetime.now())
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[QAPairStorage] ✓ Saved all Q-A pairs to {output_file}")
            logger.info(f"[QAPairStorage] Total tasks: {len(self.all_qa_data)}")
            
            return str(output_file)
        
        except Exception as e:
            logger.error(f"[QAPairStorage] Error saving combined Q-A pairs: {e}")
            return ""
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary of stored Q-A pairs"""
        total_demonstrations = sum(
            len(task.get('demonstrations', [])) 
            for task in self.all_qa_data.values()
        )
        total_qa_pairs = sum(
            len(task.get('qa_pairs', [])) 
            for task in self.all_qa_data.values()
        )
        
        return {
            'total_tasks': len(self.all_qa_data),
            'total_demonstrations': total_demonstrations,
            'total_qa_pairs': total_qa_pairs,
            'avg_demonstrations_per_task': total_demonstrations / max(len(self.all_qa_data), 1),
            'avg_qa_pairs_per_task': total_qa_pairs / max(len(self.all_qa_data), 1)
        }

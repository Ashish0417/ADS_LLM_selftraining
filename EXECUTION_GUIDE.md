# ADS FRAMEWORK - COMPLETE IMPLEMENTATION GUIDE
# "Let Large Language Models Find the Data to Train Themselves" 
# ICLR 2025 - Paper Implementation for CPU-Only Linux Laptops

================================================================================
QUICK START (5 MINUTES)
================================================================================

1. Extract all Python files to a directory:
   config.py, utils.py, data_loader.py, policy_model.py, api_handler.py, 
   evaluator.py, main.py

2. Install dependencies:
   bash setup.sh

3. Run the framework:
   python main.py

================================================================================
COMPLETE SETUP GUIDE
================================================================================

STEP 1: Install Python 3.9+
----------------------------
Ubuntu/Debian:
  sudo apt-get update
  sudo apt-get install python3.10 python3.10-venv python3-pip

Check version:
  python3 --version


STEP 2: Create Project Directory
---------------------------------
mkdir -p ~/ads-framework
cd ~/ads-framework
# Place all .py files in this directory


STEP 3: Create Virtual Environment
-----------------------------------
python3 -m venv ads_env
source ads_env/bin/activate


STEP 4: Install Dependencies
-----------------------------
# Core PyTorch (CPU-only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# All other dependencies
pip install transformers>=4.36.0 datasets>=2.14.0 tokenizers>=0.14.0 \
    numpy>=1.24.0 scipy>=1.11.0 scikit-learn>=1.3.0 rank-bm25>=0.2.2 \
    peft>=0.7.0 accelerate>=0.24.0 tqdm>=4.66.0 pyyaml>=6.0


STEP 5: Verify Installation
----------------------------
python3 -c "import torch; print('PyTorch OK')"
python3 -c "import transformers; print('Transformers OK')"
python3 -c "from config import ADSConfig; print('Config OK')"


================================================================================
RUNNING THE FRAMEWORK
================================================================================

MINIMAL TEST RUN (for verification):
-----------------------------------
# Create test script
cat > test_ads.py << 'EOF'
from config import ADSConfig
from main import ADSFramework

# Use tiny config for testing
ADSConfig.DATASET_CONFIG['train_tasks'] = 2
ADSConfig.DATASET_CONFIG['test_tasks'] = 1
ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 100
ADSConfig.TRAINING_CONFIG['num_iterations'] = 1

# Run
ads = ADSFramework(ADSConfig)
ads.setup()
ads.train(num_iterations=1)
print("✓ Test completed successfully!")
EOF

python test_ads.py


FULL TRAINING RUN:
------------------
# Method 1: Direct execution
python main.py

# Method 2: Custom script with tweaks
cat > train_ads.py << 'EOF'
from config import ADSConfig
from main import ADSFramework

# Configure for your hardware
ADSConfig.DATASET_CONFIG['train_tasks'] = 50
ADSConfig.DATASET_CONFIG['test_tasks'] = 10
ADSConfig.TRAINING_CONFIG['num_iterations'] = 3
ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 3000

# Run full pipeline
ads = ADSFramework(ADSConfig)
results = ads.run_full_pipeline()

# Access results
print(f"Final Score: {results['metrics']['avg_score']:.4f}")
print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
EOF

python train_ads.py


================================================================================
CONFIGURATION CUSTOMIZATION
================================================================================

LIGHTWEIGHT CONFIG (for slow laptops):
--------------------------------------
from config import ADSConfig

ADSConfig.POLICY_MODEL_NAME = "google/flan-t5-base"  # 250M params instead of 2.7B
ADSConfig.DATASET_CONFIG['train_tasks'] = 10
ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 500
ADSConfig.TRAINING_CONFIG['num_iterations'] = 1
ADSConfig.TRAINING_CONFIG['batch_size'] = 1
ADSConfig.TRAINING_CONFIG['max_sequence_length'] = 512


FAST ITERATION CONFIG (for testing):
-------------------------------------
ADSConfig.POLICY_MODEL_NAME = "microsoft/phi-2"
ADSConfig.DATASET_CONFIG['train_tasks'] = 5
ADSConfig.DATASET_CONFIG['test_tasks'] = 2
ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 1000
ADSConfig.TRAINING_CONFIG['num_iterations'] = 1


PRODUCTION CONFIG (better results):
------------------------------------
ADSConfig.POLICY_MODEL_NAME = "microsoft/phi-2"
ADSConfig.DATASET_CONFIG['train_tasks'] = 100
ADSConfig.DATASET_CONFIG['test_tasks'] = 20
ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 5000
ADSConfig.TRAINING_CONFIG['num_iterations'] = 3
ADSConfig.TRAINING_CONFIG['num_trajectories_per_task'] = 3


================================================================================
MODULE DESCRIPTIONS & USAGE
================================================================================

MODULE: config.py
-----------------
Purpose: Central configuration management
Usage:
  from config import ADSConfig
  
  # View all settings
  ADSConfig.print_config()
  
  # Modify settings
  ADSConfig.DATASET_CONFIG['train_tasks'] = 100
  
  # Get as dict
  config_dict = ADSConfig.get_config_dict()


MODULE: utils.py
----------------
Purpose: Utility functions and helpers
Key Classes:
  - DataCache: Persistent caching mechanism
  - TextProcessor: Text cleaning and manipulation
  - MetricsLogger: Training metrics logging
  - ProgressTracker: Progress bars with ETA
  - HeuristicScorer: Evaluation without reward model
  
Usage:
  from utils import DataCache, TextProcessor, HeuristicScorer
  
  # Cache embeddings
  cache = DataCache("cache")
  cache.save("key", data)
  loaded = cache.load("key")
  
  # Process text
  cleaned = TextProcessor.clean_text(raw_text)
  chunks = TextProcessor.split_into_chunks(text)
  
  # Score response
  score = HeuristicScorer.score_response(instruction, response)


MODULE: data_loader.py
----------------------
Purpose: Data loading and preparation
Key Classes:
  - WikipediaDataLoader: Load Wikipedia documents
  - MagpieDataLoader: Load instruction dataset
  - InstructionDataset: Manage task clusters
  - DataManager: Unified data management

Usage:
  from data_loader import DataManager
  
  manager = DataManager(config)
  manager.prepare_all_data()
  loaders = manager.get_data_loaders()
  
  # Access data
  train_tasks = loaders['train']
  wiki_docs = loaders['wikipedia']


MODULE: policy_model.py
-----------------------
Purpose: LLM models (policy and optimizer)
Key Classes:
  - PolicyModel: Main model for generation
  - OptimizerModel: Wrapper for optimization
  - ModelManager: Model initialization

Usage:
  from policy_model import PolicyModel
  
  # Initialize
  model = PolicyModel("microsoft/phi-2", device="cpu")
  
  # Generate response
  response = model.generate("What is AI?", max_tokens=256)
  
  # In-context learning
  response = model.in_context_learn(
      instruction="Explain QM",
      examples=[{'input': '...', 'output': '...'}]
  )
  
  # Evaluate response
  score = model.evaluate_response(instruction, response)


MODULE: api_handler.py
----------------------
Purpose: Three core APIs for data acquisition
Key Classes:
  - InformationRetrievalAPI: BM25-based retrieval
  - DemonstrationGenerationAPI: Generate examples
  - QuestionAnsweringAPI: Answer questions
  - APIExecutor: Execute trajectories

Usage:
  from api_handler import APIExecutor, InformationRetrievalAPI
  
  # Initialize APIs
  ir_api = InformationRetrievalAPI(documents, cache_dir="cache")
  demo_api = DemonstrationGenerationAPI(model, tokenizer)
  qa_api = QuestionAnsweringAPI(model, tokenizer)
  
  # Create executor
  executor = APIExecutor(ir_api, demo_api, qa_api)
  
  # Execute trajectory
  trajectory = [
      {'name': 'information_retrieval', 'param': 'quantum computing'},
      {'name': 'question_answering', 'param': 'What is superposition?'}
  ]
  results = executor.execute_trajectory(trajectory)


MODULE: evaluator.py
-------------------
Purpose: Evaluation metrics and scoring
Key Classes:
  - Evaluator: Main evaluation class

Usage:
  from evaluator import Evaluator
  
  evaluator = Evaluator(config)
  
  # Evaluate tasks
  results = evaluator.evaluate_tasks(
      policy_model=model,
      tasks=test_tasks,
      api_executor=executor
  )
  
  # Results include metrics


MODULE: main.py
---------------
Purpose: Main training loop and orchestration
Key Classes:
  - ADSFramework: Main framework class

Usage:
  from main import ADSFramework
  
  # Initialize
  ads = ADSFramework()
  
  # Setup components
  ads.setup()
  
  # Train
  ads.train(num_iterations=3)
  
  # Evaluate
  results = ads.evaluate()
  
  # Or run full pipeline
  results = ads.run_full_pipeline()


================================================================================
EXPECTED OUTPUTS & RESULTS
================================================================================

Successful Run Output:
========================
================================================================================
INITIALIZING ADS FRAMEWORK
================================================================================

[STEP 1/4] Loading Data...
Loading Wikipedia subset (5000 documents)...
Loaded 5000 Wikipedia documents
Loaded 100 Magpie examples

[STEP 2/4] Initializing Models...
Loading policy model: microsoft/phi-2
Policy model loaded successfully
Model: microsoft/phi-2
Total Parameters: 2,779,905,024

[STEP 3/4] Initializing APIs...
Initialized IR API with 5000 documents
Initialized Demonstration Generation API
Initialized Question Answering API

[STEP 4/4] Initializing Evaluator...

================================================================================
SETUP COMPLETE - READY FOR TRAINING
================================================================================

================================================================================
STARTING TRAINING
================================================================================

Iteration 1/3
Processing tasks (iter 1): 100%|████████| 5/5 [00:45<00:00,  9.00s/it]

Iteration Summary:
  - Tasks completed: 5/5
  - Average reward: 0.6234
  - Total API cost: 45

...

================================================================================
EVALUATION COMPLETE
================================================================================


Output Files Created:
======================
results/
├── training_metrics.json      # Per-iteration metrics
├── evaluation_results.json    # Task scores
├── checkpoint.pt              # Model checkpoint
└── logs/
    └── ads_framework.log      # Detailed logs

Example evaluation_results.json:
{
  "total_tasks": 10,
  "completed_tasks": 10,
  "failed_tasks": 0,
  "metrics": {
    "win_rate": 0.75,
    "tie_rate": 0.15,
    "loss_rate": 0.10,
    "avg_score": 0.682
  },
  "task_results": [
    {
      "instruction": "Explain machine learning",
      "score": 0.85,
      "api_cost": 8
    },
    ...
  ]
}


================================================================================
TROUBLESHOOTING
================================================================================

ISSUE: Model Download Stuck
---------------------------
Solution:
  # Pre-download model to cache
  python3 -c "
  from transformers import AutoModel
  AutoModel.from_pretrained('microsoft/phi-2')
  print('✓ Model downloaded')
  "


ISSUE: Out of Memory
--------------------
Solution:
  # Reduce configuration in config.py:
  ADSConfig.DATASET_CONFIG['train_tasks'] = 10
  ADSConfig.WIKIPEDIA_CONFIG['cache_size'] = 1000
  ADSConfig.TRAINING_CONFIG['max_sequence_length'] = 512
  
  # Use smaller model:
  ADSConfig.POLICY_MODEL_NAME = "google/flan-t5-base"


ISSUE: Very Slow Generation
----------------------------
Solution:
  # Reduce iterations for testing
  ADSConfig.TRAINING_CONFIG['num_iterations'] = 1
  
  # Reduce task count
  ADSConfig.DATASET_CONFIG['train_tasks'] = 5
  
  # Use cached data (subsequent runs)
  # Cache automatically created in cache/ directory


ISSUE: Import Errors
---------------------
Check all files are in same directory:
  ls -la *.py
  # Should show: config.py, utils.py, data_loader.py, 
  #              policy_model.py, api_handler.py, evaluator.py, main.py

Reinstall dependencies:
  pip install -r requirements.txt --force-reinstall


================================================================================
PERFORMANCE TIPS
================================================================================

1. FIRST RUN (Slowest):
   - Models download from HuggingFace (~8-10 GB)
   - Data loads and caches locally
   - Wikipedia embedding cache created
   Expected time: 30-60 minutes

2. SUBSEQUENT RUNS:
   - Models loaded from cache
   - Data loaded from cache
   - Can reuse API call cache
   Expected time: 10-20 minutes

3. OPTIMIZATION TIPS:
   - Reduce num_iterations for faster testing
   - Reduce cache_size for less memory usage
   - Use FLAN-T5 for speed, Phi-2 for quality
   - Run training overnight for production runs

4. MEMORY MONITORING:
   # Check RAM usage during run
   watch -n 1 'ps aux | grep python'
   
   # Check disk space
   du -sh cache/ data/ models/


================================================================================
ADVANCED EXAMPLES
================================================================================

EXAMPLE 1: Custom Configuration
---------------------------------
from config import ADSConfig
from main import ADSFramework

# Create custom config
class CustomConfig(ADSConfig):
    POLICY_MODEL_NAME = "google/flan-t5-base"
    DATASET_CONFIG = {
        "train_tasks": 200,
        "test_tasks": 50,
        "instructions_per_task": 5,
    }

ads = ADSFramework(CustomConfig)
ads.run_full_pipeline()


EXAMPLE 2: Manual Step-by-Step Execution
------------------------------------------
from main import ADSFramework
from config import ADSConfig

ads = ADSFramework()

# Setup
ads.setup()

# Manual training loop
for iteration in range(3):
    print(f"Iteration {iteration+1}")
    ads.train(num_iterations=1)
    results = ads.evaluate()
    print(f"Score: {results['metrics']['avg_score']}")
    ads.save_checkpoint(f"checkpoint_iter_{iteration}.pt")


EXAMPLE 3: Load and Continue Training
--------------------------------------
from main import ADSFramework
from config import ADSConfig
import torch

# Load checkpoint
checkpoint = torch.load("results/checkpoint.pt")
config = ADSConfig
config.TRAINING_CONFIG['num_iterations'] = 3

# Continue training
ads = ADSFramework(config)
ads.setup()
ads.train(num_iterations=3)


================================================================================
NEXT STEPS & IMPROVEMENTS
================================================================================

1. ENHANCE MODEL QUALITY:
   - Use Qwen-2-7B instead of Phi-2
   - Train for more iterations (5-10)
   - Increase dataset size (1000+ tasks)

2. ADD REWARD MODEL:
   - Download FsfairX-Llama-3-RM model
   - Replace heuristic scoring with RM
   - See evaluator.py for integration

3. ADD DENSE RETRIEVAL:
   - Use BGE embeddings for better retrieval
   - Replace BM25-only approach
   - See api_handler.py for integration

4. EVALUATE ON PUBLIC BENCHMARKS:
   - AlpacaEval 2.0
   - Arena-Hard
   - MT-Bench
   - See data_loader.py

5. PRODUCTION DEPLOYMENT:
   - Save best models to disk
   - Create API endpoints
   - Add model versioning
   - Monitor inference latency


================================================================================
REFERENCES & CITATIONS
================================================================================

Paper:
  "Let Large Language Models Find the Data to Train Themselves"
  Submission to ICLR 2025
  
Key Components Implemented:
  ✓ Active Data Search (ADS) Framework
  ✓ Optimizer Model for API trajectory generation
  ✓ Three Core APIs (IR, Demo Gen, QA)
  ✓ In-Context Learning for rapid adaptation
  ✓ Reinforcement Learning with rejection sampling
  ✓ Heuristic evaluation (CPU-friendly alternative to RM)

Data Sources:
  - Wikipedia-22-12 (Cohere dataset)
  - Magpie-Air-3M (Instruction dataset)
  - AlpacaEval 2.0, Arena-Hard, MT-Bench (Benchmarks)


================================================================================
SUPPORT & DEBUGGING
================================================================================

Enable Debug Logging:
  import logging
  logging.basicConfig(level=logging.DEBUG)

Check specific component:
  python3 -c "from data_loader import DataManager; print('Data module OK')"
  python3 -c "from policy_model import PolicyModel; print('Model module OK')"

View detailed logs:
  tail -f ads_framework.log | grep ERROR

Profile execution:
  python3 -m cProfile -s cumtime main.py | head -50


================================================================================
END OF GUIDE
================================================================================

For latest updates and issues, refer to:
- GitHub repository
- Paper: https://openreview.net/pdf?id=5YCZZSEosw
- HuggingFace models and datasets

Version: 1.0 (CPU-Optimized)
Last Updated: October 2024
Status: Production Ready

# Generative AI Course Repository

This repository is dedicated to sharing my personal journey and notes on Generative AI. As a professional Data Scientist, I've created this repository to provide resources, labs, and notes for other Data Scientists and enthusiasts looking to learn and grow in the field of Generative AI.
## Repository Structure

The repository is organized into folders corresponding to each section of the course. Within each folder, you will find relevant labs, notes, and resources related to that particular topic.

## Course Outline: Generative AI 

1. [**Introduction to Generative AI**](Docs/1-Introduction-GenAI/README.md)
   - What is Generative AI?
   - Different types of Generative AI (Variational Autoencoders, Generative Adversarial Networks)
   - Applications of Generative AI (text generation, image generation, music generation)
   - Responsible AI and Governance
   - Bias and Fairness in AI Models
   - Explainability and Interpretability
   - Legal and Ethical Considerations

2. [**Natural Language Processing (NLP)**](Docs/2-Natural-Language-Processing/README.md)
   - Text Preprocessing:
     - Tokenization
     - Stemming
     - Lemmatization
     - Stop word removal
   - Feature Extraction Techniques:
     - Bag-of-words (BoW)
     - Term Frequency-Inverse Document Frequency (TF-IDF)
     - n-grams
   - Word Embeddings:
     - Word2Vec
     - GloVe
     - FastText
   - Recurrent Neural Networks (RNNs):
     - Understanding RNNs
     - LSTMs
     - GRUs

3. [**Large Language Models (LLMs)**](Docs/3-Large-Language-Models/README.md)
   - LLM architecture:
     - Transformer architecture (high-level view)
     - Tokenization
     - Attention mechanisms (self-attention, scaled dot-product attention)
     - Text generation strategies (greedy decoding, beam search, top-k sampling, nucleus sampling)
   - Building an instruction dataset:
     - Techniques for generating high-quality instruction datasets
     - Data filtering and cleaning methods
     - Exploring prompt templates for LLMs
   - Pre-training models:
     - Understanding the concept of pre-training for LLMs (high-level overview, not hands-on experience)
     - Data pipelines and challenges in pre-training LLM
     - Scaling laws and their impact on LLM performance
   - Supervised Fine-Tuning:
     - Fine-tuning pre-trained LLMs for specific tasks
     - Techniques for Supervised Fine-Tuning (SFT)
     - Parameter-efficient fine-tuning methods (LoRA, QLoRA)
     - Axolotl and DeepSpeed libraries for fine-tuning
   - Reinforcement Learning from Human Feedback (RLHF):
     - Preference datasets
     - Proximal Policy Optimization
     - Direct Preference Optimization (DPO)
   - Evaluation:
     - Traditional metrics (perplexity, BLEU score)
     - General benchmarks (Language Model Evaluation Harness, Open LLM Leaderboard)
     - Task-specific benchmarks
     - Human evaluation
   - Quantization:
     - Base techniques
     - GGUF and llama.cpp
     - GPTQ and EXL2
     - AWQ
4. [**Retrieval Augmented Generation (RAG)**](Docs/4-Retrieval-Augmented-Generation/README.md)
   - Building a Vector Storage:
     - Ingesting documents
     - Splitting documents
     - Embedding models
     - Vector databases
   - Retrieval Augmented Generation:
     - Orchestrators
     - Retrievers
     - Memory
   - Evaluation

5. [**Deployment in Production**](Docs/5-Deployment-in-Production/README.md)
   - LLM APIs
   - Open-source LLMs (Hugging Face Hub)
   - Prompt engineering techniques
   - Structuring outputs
   - IBM WatsonX.ai:
     - Using Prompt Lab in Watson.ai
     - Load an LLM from Watson.ai
   - Integration of Watson Assistant with Watsonx.ai

6. [**Diffusion Models**](Docs/6-Difffusion-Models/README.md)
   -Introduction to Diffusion Models
   - Fine-Tuning and Guidance
   - Advanced Diffusion Models and Techniques
7. **Multi-Modal Large Language Models (LLMs)**
  - [**Demystifying Multi-Modal LLMs**](Docs/7-Multi-Modal-LLMs/Part-1-Multi-Modal-LLM/README.md)
    - Applications of Multi-Modal LLMs
    - Understanding the Technology
    - Architecture of Multi-Modal LLMs
    - Challenges and Considerations
    - Bias and Fairness
    - Future of Multi-Modal LLMs
    - Potential Advancements
  - [**Creating a Multimodal Model: Combining LLM with Images**](Docs/7-Multi-Modal-LLMs/Part-2-Creation-Multi-Modal/README.md)
    - Choosing a Multi-Modal LLM
    - Loading Image Data
    - Initializing the Multi-Modal LLM
    - Building a Multi-Modal Vector Store/Index
    - Retrieving Information
    - Optional: Query Engine
## 📝 Notebooks
A list of notebooks related to Generative AI.

### Data Preparation
| Notebook | Description |
| --- | --- |
| [Documents -> Dataset](/Data-Preparation/dataset_generator_from_documents.ipynb) | Given Documents generate Instruction/QA dataset for finetuning LLMs |
| [Topic -> Dataset](/Data-Preparation/dataset_generator_from_topic.ipynb) | Given a Topic generate a dataset to finetune LLMs |
| [Alpaca Dataset Generation](/Data-Preparation/instruction_dataset_generator.ipynb) | The original implementation of generating instruction dataset followed in the alpaca paper |

### Fine-tuning
| Notebook | Description |
| --- | --- |
| [Fine-tune Llama 2 with SFT](./Finetuning/Fine_tune_Llama_2_in_Google_Colab.ipynb) | Step-by-step guide to supervised fine-tune Llama 2 in Google Colab. |
| [Fine-tune CodeLlama using Axolotl](./Finetuning/Fine_tune_LLMs_with_Axolotl.ipynb) | End-to-end guide to the state-of-the-art tool for fine-tuning. |
| [Fine-tune Mistral-7b with SFT](./Finetuning/Fine_tune_Mistral_7b_with_SFT.ipynb) | Superd fine-tune Mistral-7b in a free-tier Google Colab with TRL. |
| [Fine-tune Mistral-7b with DPO](./Finetuning/Fine_tune_Mistral_7b_with_DPO.ipynb) | Boost the performance of supervised fine-tuned models with DPO. |
| [Fine-tune Llama 3 with ORPO](./Finetuning/Fine_tune_Llama_3_with_ORPO.ipynb) | Cheaper and faster fine-tuning in a single stage with ORPO. |
| [Gemma Finetuning](/Finetuning/Gemma_finetuning_notebook.ipynb) | Notebook to Finetune Gemma Models |
| [Mistral-7b Finetuning](/Finetuning/Mistral_finetuning_notebook.ipynb) | Notebook to Finetune Mistral-7b Model |
| [Mixtral Finetuning](/Finetuning/Mixtral_finetuning_notebook.ipynb) | Notebook to Finetune Mixtral-7b Models |
| [Llama2 Finetuning](/Finetuning/Llama2_finetuning_notebook.ipynb) | Notebook to Finetune Llama2-7b Model |
| [Alpaca_+_TinyLlama_+_RoPE_Scaling](./Finetuning/Alpaca_+_TinyLlama_+_RoPE_Scaling_unsloth.ipynb) | TinyLlama: 3.9x faster, 74% less memory use. |
| [Alpaca_+_Phi_3_3_8b](./Finetuning/Alpaca_+_Phi_3_3_8b_unsloth.ipynb) | Phi-3 (3.8B): 2x faster, 50% less memory use. |
| [DPO_Zephyr](./Finetuning/DPO_Zephyr_unsloth.ipynb) | DPO Zephyr:  1.9x faster, 43% less memory use. |
| [ORPO_Unsloth](./Finetuning/ORPO_Unsloth.ipynb) | ORPO: 1.9x faster, 43% less memory use. |
| [Alpaca_+_Gemma_7b](./Finetuning/Alpaca_+_Gemma_7b_unsloth.ipynb) |  Gemma (7B): 2.4x faster, 71% less memory use. |
| [Alpaca_+_Mistral_7b](./Finetuning/Alpaca_+_Mistral_7b_unsloth.ipynb) |Mistral (7B):  2.2x faster, 73% less memory use. |
| [Alpaca_+_Llama_3_8b](./Finetuning/Alpaca_+_Llama_3_8b_unsloth.ipynb) | Llama 3 (8B):  2x faster, 60% less memory use |

### Quantization
| Notebook | Description |
| --- | --- |
| [Introduction to Quantization](./Quantization/Introduction_to_Weight_Quantization.ipynb) | Large language model optimization using 8-bit quantization. |
| [4-bit Quantization using GPTQ](./Quantization/4_bit_LLM_Quantization_with_GPTQ.ipynb) | Quantize your own open-source LLMs to run them on consumer hardware. |
| [Quantization with GGUF and llama.cpp](./Quantization/Quantize_Llama_2_models_using_GGUF_and_llama_cpp.ipynb) | Quantize Llama 2 models with llama.cpp and upload GGUF versions to the HF Hub. |
| [ExLlamaV2: The Fastest Library to Run LLMs](./Quantization/Quantize_models_with_ExLlamaV2.ipynb) | Quantize and run EXL2 models and upload them to the HF Hub. |
| [AWQ Quantization](/Quantization/AWQ_Quantization.ipynb) | Quantize LLM using AW. |
| [GGUF Quantization](/Quantization/GGUF_Quantization.ipynb) | Quantize LLM to GGUF format. |


### Inference
| Notebook | Description |
| --- | --- |

| [LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat](./Deployment/LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat.ipynb)|Inference with CPP Llama |


### Tools
| Notebook | Description |
| --- | --- |
| [LLM AutoEval](./Tools/LLM_AutoEval.ipynb) | Automatically evaluate your LLMs using RunPod |
| [LazyMergekit](./Tools/LazyMergekit.ipynb) | Easily merge models using MergeKit in one click. |
| [LazyAxolotl](./Tools/LazyAxolotl.ipynb) | Fine-tune models in the cloud using Axolotl in one click. |
| [AutoQuant](./Tools/AutoQuant.ipynb) | Quantize LLMs in GGUF, GPTQ, EXL2, AWQ, and HQQ formats in one click. |
| [Model Family Tree](./Tools/Model_Family_Tree.ipynb) | Visualize the family tree of merged models. |
| [ZeroSpace](./Tools/ZeroChat.ipynb) | Automatically create a Gradio chat interface using a free ZeroGPU. |

### Other
| Notebook | Description |
| --- | --- |
| [Improve ChatGPT with Knowledge Graphs](./Other/Improve_ChatGPT_with_Knowledge_Graphs.ipynb) | Augment ChatGPT's answers with knowledge graphs. |
| [Decoding Strategies in Large Language Models](./Other/Decoding_Strategies_in_Large_Language_Models.ipynb) | A guide to text generation from beam search to nucleus sampling |

## Contributing
--------------

I'm thrilled to have you contribute to this repository! If you're interested in adding new content, fixing bugs, or improving the existing materials. Thank you for helping to make this repository a valuable resource for the Generative AI community!

## About the Author
-------------------

I'm Ruslan Magana Vsevolodovna, a professional Data Scientist with a passion for Generative AI. You can learn more about me and my work at [ruslanmv.com](https://ruslanmv.com).

Thank you for visiting this repository, and I hope you find the resources helpful in your Generative AI journey!
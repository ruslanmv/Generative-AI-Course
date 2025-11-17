# Generative AI Course - Production-Ready Educational Platform

<div align="center">

![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-orange)

**A comprehensive, commercial-grade platform for mastering Generative AI, Large Language Models, and Multimodal Systems**

[What This Offers](#-what-this-platform-offers) •
[Installation](#-installation) •
[Notebooks](#-interactive-notebooks) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation)

</div>

---

## 🌟 What This Platform Offers

This **Generative AI Course** is your complete, production-ready toolkit for mastering modern AI. Whether you're a beginner or an experienced practitioner, this platform provides everything you need:

### 🎓 For Learners
- **177+ Hands-On Jupyter Notebooks**: Interactive tutorials covering every aspect of Generative AI
- **Step-by-Step Guides**: From basic NLP to advanced multimodal models
- **Real-World Examples**: Practical implementations you can use immediately
- **Progressive Learning Path**: Organized curriculum from fundamentals to advanced topics

### 💼 For Practitioners
- **Production-Ready Code**: Deploy LLMs with our FastAPI inference server
- **Fine-Tuning Recipes**: Train Llama, Mistral, Gemma, and more with LoRA/QLoRA
- **Optimization Techniques**: Quantization (4-bit/8-bit), model compression, efficient inference
- **Enterprise Tools**: Automated build system, testing, CI/CD integration

### 🚀 For Businesses
- **Commercial License**: Apache 2.0 - free for commercial use
- **Deployment Solutions**: Production servers with streaming, health checks, monitoring
- **Scalable Architecture**: Multi-worker deployment, load balancing, Docker-ready
- **Professional Standards**: PEP 8 compliant, type-hinted, fully documented

### 🛠️ What You'll Build
- Fine-tune Large Language Models for your specific use case
- Deploy production LLM inference servers with streaming responses
- Create multimodal AI applications combining vision and language
- Build RAG (Retrieval Augmented Generation) systems
- Optimize models for consumer hardware with quantization
- Implement diffusion models for image generation

---

## 📖 About

The **Generative AI Course** is a production-ready educational platform designed for Data Scientists, ML Engineers, and AI enthusiasts to master the complete spectrum of Generative AI technologies. This repository combines **theoretical foundations** with **hands-on implementation**, offering 177+ Jupyter notebooks, production-ready Python modules, and deployment solutions for real-world applications.

### What Makes This Different?

- **Production-Ready Code**: All modules follow industry best practices with comprehensive type hints, docstrings, and error handling
- **Modern Development Stack**: Built with `uv` package manager, automated testing, and CI/CD-ready infrastructure
- **Comprehensive Coverage**: From fundamentals to advanced topics including LLMs, diffusion models, multimodal systems, and deployment
- **Hands-On Learning**: 177+ interactive notebooks covering every aspect of Generative AI
- **Enterprise Deployment**: Production FastAPI servers, quantization, and optimization techniques
- **Open Source**: Apache 2.0 licensed for commercial and educational use

---

## 🚀 Installation

### Simple 3-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/ruslanmv/Generative-AI-Course.git
cd Generative-AI-Course

# 2. Install Jupyter and core dependencies
pip install jupyter notebook ipykernel

# 3. Start exploring!
jupyter notebook
```

### For Production Use (Recommended)

If you want to use the production FastAPI server and development tools:

```bash
# Install uv package manager (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
make install-all

# Or manually with pip
pip install -e ".[all]"
```

### 📝 Important Note About Notebook Dependencies

**Each notebook may require specific dependencies** based on the model or technique it demonstrates:

- **Most notebooks are self-contained** and will install their required packages in the first cell
- **Some notebooks require GPU** with CUDA support (check the notebook introduction)
- **Cloud notebooks** (marked with Colab badges) can run for free on Google Colab
- **Large models** may need 16GB+ RAM or cloud resources

**Before running a notebook:**
1. Open the notebook and check the first few cells for dependency installation
2. Run the pip install commands provided in the notebook
3. Some notebooks include `!pip install package_name` - these will auto-install

**Common notebook-specific dependencies:**
- Fine-tuning notebooks: `transformers`, `peft`, `trl`, `bitsandbytes`
- Quantization notebooks: `auto-gptq`, `llama-cpp-python`, `exllamav2`
- Multimodal notebooks: `weaviate-client`, `opencv-python`, `scikit-image`
- Diffusion notebooks: `diffusers`, `controlnet_aux`

💡 **Tip**: Start with the notebooks in order - early notebooks often explain dependency installation for the entire section.

---

## 📓 Interactive Notebooks

All notebooks are **directly accessible** by clicking the links below. Each notebook is **self-contained** with installation instructions.

### 📊 Data Preparation (17 Notebooks)

Create custom datasets for fine-tuning LLMs from documents, topics, or existing data.

| Notebook | Description | Link |
|----------|-------------|------|
| **Documents → Dataset** | Generate instruction/QA datasets from your documents | [Open Notebook](Data-Preparation/dataset_generator_from_documents.ipynb) |
| **Topic → Dataset** | Create datasets from a topic or domain | [Open Notebook](Data-Preparation/dataset_generator_from_topic.ipynb) |
| **Alpaca Dataset Generator** | Original Alpaca-style instruction dataset creation | [Open Notebook](Data-Preparation/instruction_dataset_generator.ipynb) |
| **News Classification Dataset** | Create classification datasets with GPT-3.5 | [Open Notebook](Data-Preparation/Creating_News_Classification_Instruction_Dataset_using_GPT3_5.ipynb) |
| **Translation Dataset (Bilingual)** | Prepare bilingual translation datasets | [Open Notebook](Data-Preparation/ambari/bilingual_dataset_prep.ipynb) |
| **Translation Dataset (Multi-lang)** | Translation between multiple languages | [Open Notebook](Data-Preparation/ambari/translation_between_languages.ipynb) |
| **Romanized Indic Languages** | Translation to romanized Indic languages | [Open Notebook](Data-Preparation/ambari/translation_to_romanised_indic_languages.ipynb) |
| **DPO Dataset Formatting** | Format datasets for Direct Preference Optimization | [Open Notebook](Data-Preparation/ambari/DPO_formate.ipynb) |
| **DPO Translation Dataset** | Create DPO datasets for translation tasks | [Open Notebook](Data-Preparation/ambari/DPO_translation.ipynb) |
| **Instruction Dataset Prep** | Prepare instruction-following datasets | [Open Notebook](Data-Preparation/ambari/dataset_prep_instruction.ipynb) |
| **Create Custom Dataset** | General dataset creation utilities | [Open Notebook](Data-Preparation/ambari/create_dataset.ipynb) |
| **Translation Dataset Creator** | Build translation training datasets | [Open Notebook](Data-Preparation/ambari/create_translation_dataset.ipynb) |
| **Translation Inference** | Test translation models on single examples | [Open Notebook](Data-Preparation/ambari/translate_inference_single.ipynb) |
| **Combine Translations** | Merge and combine translation datasets | [Open Notebook](Data-Preparation/ambari/final_translation_combine.ipynb) |
| **Kannada Dataset Merger** | Merge Kannada language datasets | [Open Notebook](Data-Preparation/kannada/merge_dataset.ipynb) |
| **LLaVA Image Prep** | Prepare image datasets for LLaVA training | [Open Notebook](Data-Preparation/llava/image_prep_dataset.ipynb) |
| **Aarogya Formatting** | Format Aarogya medical datasets | [Open Notebook](Data-Preparation/aarogya/formatting.ipynb) |

---

### 🔧 Fine-Tuning LLMs (17 Notebooks)

Master parameter-efficient fine-tuning with LoRA, QLoRA, SFT, DPO, and ORPO.

| Notebook | Model | Technique | Framework | Link |
|----------|-------|-----------|-----------|------|
| **Llama 2 SFT (Google Colab)** | Llama 2 7B | SFT | Transformers | [Open Notebook](Finetuning/Fine_tune_Llama_2_in_Google_Colab.ipynb) |
| **Llama 2 QLora** | Llama 2 7B | QLora | PEFT | [Open Notebook](LLMs/LLama2/Llama_2_Fine_Tuning_using_QLora.ipynb) |
| **Llama 3 ORPO** | Llama 3 8B | ORPO | TRL | [Open Notebook](Finetuning/Fine_tune_Llama_3_with_ORPO.ipynb) |
| **Llama 3 Alpaca (Unsloth)** | Llama 3 8B | SFT | Unsloth | [Open Notebook](Finetuning/Alpaca_+_Llama_3_8b_unsloth.ipynb) |
| **Mistral 7B SFT** | Mistral 7B | SFT | TRL | [Open Notebook](Finetuning/Fine_tune_Mistral_7b_with_SFT.ipynb) |
| **Mistral 7B DPO** | Mistral 7B | DPO | TRL | [Open Notebook](Finetuning/Fine_tune_Mistral_7b_with_DPO.ipynb) |
| **Mistral 7B (Unsloth)** | Mistral 7B | SFT | Unsloth | [Open Notebook](Finetuning/Alpaca_+_Mistral_7b_unsloth.ipynb) |
| **Mistral Fine-tuning** | Mistral 7B | SFT | Transformers | [Open Notebook](Finetuning/Mistral_finetuning_notebook.ipynb) |
| **Gemma SFT** | Gemma 7B | SFT | Transformers | [Open Notebook](LLMs/Gemma/finetune-gemma.ipynb) |
| **Gemma (Unsloth)** | Gemma 7B | SFT | Unsloth | [Open Notebook](Finetuning/Alpaca_+_Gemma_7b_unsloth.ipynb) |
| **Gemma Fine-tuning** | Gemma 2B/7B | SFT | Transformers | [Open Notebook](Finetuning/Gemma_finetuning_notebook.ipynb) |
| **Mixtral Fine-tuning** | Mixtral 8x7B | SFT | Transformers | [Open Notebook](Finetuning/Mixtral_finetuning_notebook.ipynb) |
| **Phi-3 (Unsloth)** | Phi-3 3.8B | SFT | Unsloth | [Open Notebook](Finetuning/Alpaca_+_Phi_3_3_8b_unsloth.ipynb) |
| **TinyLlama + RoPE** | TinyLlama | SFT + RoPE | Unsloth | [Open Notebook](Finetuning/Alpaca_+_TinyLlama_+_RoPE_Scaling_unsloth.ipynb) |
| **DPO Zephyr** | Zephyr | DPO | Unsloth | [Open Notebook](Finetuning/DPO_Zephyr_unsloth.ipynb) |
| **ORPO (Unsloth)** | Various | ORPO | Unsloth | [Open Notebook](Finetuning/ORPO_Unsloth.ipynb) |
| **Axolotl Framework** | CodeLlama | SFT | Axolotl | [Open Notebook](Finetuning/Fine_tune_LLMs_with_Axolotl.ipynb) |

**Performance Gains**: Unsloth notebooks achieve up to **3.9x faster training** and **74% less memory usage** compared to standard implementations.

---

### ⚡ Quantization (6 Notebooks)

Compress models to run on consumer hardware without significant quality loss.

| Notebook | Format | Description | Link |
|----------|--------|-------------|------|
| **Introduction to Quantization** | 8-bit | Foundation of weight quantization | [Open Notebook](Quantization/Introduction_to_Weight_Quantization.ipynb) |
| **GPTQ Quantization** | GPTQ | 4-bit quantization with GPTQ | [Open Notebook](Quantization/4_bit_LLM_Quantization_with_GPTQ.ipynb) |
| **GGUF & llama.cpp** | GGUF | Quantize for llama.cpp inference | [Open Notebook](Quantization/Quantize_Llama_2_models_using_GGUF_and_llama_cpp.ipynb) |
| **GGUF Quantization** | GGUF | General GGUF quantization guide | [Open Notebook](Quantization/GGUF_Quantization.ipynb) |
| **AWQ Quantization** | AWQ | Activation-aware weight quantization | [Open Notebook](Quantization/AWQ_Quantization.ipynb) |
| **ExLlamaV2** | EXL2 | Fastest inference with ExLlamaV2 | [Open Notebook](Quantization/Quantize_models_with_ExLlamaV2.ipynb) |

---

### 🚀 Deployment & Inference (5 Notebooks)

Production-ready deployment strategies and performance benchmarking.

| Notebook | Technology | Description | Link |
|----------|-----------|-------------|------|
| **Llama.cpp Python** | llama.cpp | Inference with llama.cpp Python bindings | [Open Notebook](Deployment/LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat.ipynb) |
| **vLLM Benchmark** | vLLM | Benchmark vLLM performance | [Open Notebook](Deployment/vLLM_benchmark.ipynb) |
| **TGI vs vLLM** | TGI, vLLM | Compare Text Generation Inference vs vLLM | [Open Notebook](Deployment/TGI_VLLM_inference.ipynb) |
| **Streaming Inference** | Hugging Face | Implement streaming text generation | [Open Notebook](utils/streaming_inference_hf.ipynb) |
| **FastAPI Server** | FastAPI | Production server (see `/Deployment/server.py`) | [View Code](Deployment/server.py) |

---

### 🎨 Diffusion Models (7 Notebooks)

Generate and manipulate images with state-of-the-art diffusion techniques.

| Notebook | Technique | Description | Link |
|----------|-----------|-------------|------|
| **Negative Prompts** | Stable Diffusion | Control generation with negative prompts | [Open Notebook](Docs/6-Difffusion-Models/6.1-NegativePrompt.ipynb) |
| **Image-to-Image** | img2img | Transform images with diffusion | [Open Notebook](Docs/6-Difffusion-Models/6.2-img2img.ipynb) |
| **Image Interpolation** | Interpolation | Smooth transitions between images | [Open Notebook](Docs/6-Difffusion-Models/6.3-imageinterpolation.ipynb) |
| **DiffEdit v1** | DiffEdit | Semantic image editing | [Open Notebook](Docs/6-Difffusion-Models/6.4-DiffEdit.ipynb) |
| **DiffEdit v2** | DiffEdit | Enhanced image editing | [Open Notebook](Docs/6-Difffusion-Models/6.4-DiffEdit_v2.ipynb) |
| **DiffEdit v3** | DiffEdit | Advanced editing techniques | [Open Notebook](Docs/6-Difffusion-Models/6.4-DiffEdit_v3.ipynb) |
| **DiffEdit v4** | DiffEdit | Latest DiffEdit improvements | [Open Notebook](Docs/6-Difffusion-Models/6.4-DiffEdit_v4.ipynb) |

---

### 🖼️ Multimodal AI (6 Notebooks)

Combine vision, language, and other modalities for advanced AI applications.

| Notebook | Topic | Description | Link |
|----------|-------|-------------|------|
| **Overview of Multimodality** | Introduction | Fundamentals of multimodal AI | [Open Notebook](Multimodal/L1_Overview_of_Multimodality.ipynb) |
| **Multimodal Search** | Search | Cross-modal search and retrieval | [Open Notebook](Multimodal/L2_Multimodal_Search.ipynb) |
| **Large Multimodal Models** | LMMs | Understanding LMMs like GPT-4V | [Open Notebook](Multimodal/L3_LMMs.ipynb) |
| **Multimodal RAG** | RAG | Retrieval with images and text | [Open Notebook](Multimodal/L4_Multimodal_RAG.ipynb) |
| **Industry Applications** | Applications | Real-world multimodal use cases | [Open Notebook](Multimodal/L5_Industry_Applications.ipynb) |
| **Multimodal Recommender** | Recommendation | Build multimodal recommendation systems | [Open Notebook](Multimodal/L6_Multimodal_Recommender.ipynb) |

---

### 🛠️ Tools & Utilities (6 Notebooks)

Automate common tasks with one-click tools.

| Notebook | Tool | Description | Link |
|----------|------|-------------|------|
| **AutoQuant** | Quantization | Quantize to GPTQ, GGUF, AWQ, EXL2, HQQ in one click | [Open Notebook](Tools/AutoQuant.ipynb) |
| **LazyMergekit** | Model Merging | Merge models using MergeKit easily | [Open Notebook](Tools/LazyMergekit.ipynb) |
| **LazyAxolotl** | Fine-tuning | Fine-tune in the cloud with Axolotl | [Open Notebook](Tools/LazyAxolotl.ipynb) |
| **LLM AutoEval** | Evaluation | Automatically evaluate LLMs with RunPod | [Open Notebook](Tools/LLM_AutoEval.ipynb) |
| **Model Family Tree** | Visualization | Visualize merged model genealogy | [Open Notebook](Tools/Model_Family_Tree.ipynb) |
| **ZeroChat** | Deployment | Create Gradio chat UI with ZeroGPU | [Open Notebook](Tools/ZeroChat.ipynb) |

---

### 🧬 Transformers Advanced (118+ Notebooks)

Comprehensive collection covering every aspect of transformer-based models.

#### NLP Tasks

| Notebook | Task | Link |
|----------|------|------|
| **Text Classification** | Classification | [Open Notebook](Transformers/text_classification.ipynb) |
| **Text Classification (TF)** | Classification | [Open Notebook](Transformers/text_classification-tf.ipynb) |
| **Token Classification** | NER, POS tagging | [Open Notebook](Transformers/token_classification.ipynb) |
| **Question Answering** | QA | [Open Notebook](Transformers/question_answering.ipynb) |
| **Summarization** | Summarization | [Open Notebook](Transformers/summarization.ipynb) |
| **Translation** | Machine Translation | [Open Notebook](Transformers/translation.ipynb) |
| **Language Modeling** | LM | [Open Notebook](Transformers/language_modeling.ipynb) |
| **Language Modeling from Scratch** | LM | [Open Notebook](Transformers/language_modeling_from_scratch.ipynb) |

#### Computer Vision

| Notebook | Task | Link |
|----------|------|------|
| **Image Classification** | Classification | [Open Notebook](Transformers/image_classification.ipynb) |
| **Image Similarity** | Similarity | [Open Notebook](Transformers/image_similarity.ipynb) |
| **Image Captioning (BLIP)** | Captioning | [Open Notebook](Transformers/image_captioning_blip.ipynb) |
| **Image Captioning (Pix2Struct)** | Captioning | [Open Notebook](Transformers/image_captioning_pix2struct.ipynb) |
| **Semantic Segmentation** | Segmentation | [Open Notebook](Transformers/semantic_segmentation.ipynb) |
| **Segment Anything** | Segmentation | [Open Notebook](Transformers/segment_anything.ipynb) |
| **Zero-Shot Object Detection** | Detection | [Open Notebook](Transformers/zeroshot_object_detection_with_owlvit.ipynb) |
| **Video Classification** | Video | [Open Notebook](Transformers/video_classification.ipynb) |

#### Audio & Speech

| Notebook | Task | Link |
|----------|------|------|
| **Audio Classification** | Classification | [Open Notebook](Transformers/audio_classification.ipynb) |
| **Speech Recognition** | ASR | [Open Notebook](Transformers/speech_recognition.ipynb) |
| **Multi-Lingual Speech Recognition** | ASR | [Open Notebook](Transformers/multi_lingual_speech_recognition.ipynb) |

#### Time Series & Specialized

| Notebook | Domain | Link |
|----------|--------|------|
| **Time Series Transformers** | Forecasting | [Open Notebook](Transformers/time-series-transformers.ipynb) |
| **Multivariate Informer** | Forecasting | [Open Notebook](Transformers/multivariate_informer.ipynb) |
| **Protein Folding** | Biology | [Open Notebook](Transformers/protein_folding.ipynb) |
| **Protein Language Modeling** | Biology | [Open Notebook](Transformers/protein_language_modeling.ipynb) |
| **DNA Sequence Modeling** | Biology | [Open Notebook](Transformers/nucleotide_transformer_dna_sequence_modelling.ipynb) |
| **Annotated Diffusion** | Diffusion | [Open Notebook](Transformers/annotated_diffusion.ipynb) |

#### Optimization & Deployment

| Notebook | Topic | Link |
|----------|-------|------|
| **ONNX Export** | Deployment | [Open Notebook](Transformers/onnx-export.ipynb) |
| **Benchmark** | Performance | [Open Notebook](Transformers/benchmark.ipynb) |
| **TPU Training (TF)** | Training | [Open Notebook](Transformers/tpu_training-tf.ipynb) |

[**View all 118+ Transformer notebooks →**](Transformers/)

---

### 📚 Transformers Basics (20 Notebooks)

Foundation tutorials for getting started with transformers.

| Notebook | Topic | Link |
|----------|-------|------|
| **How to Train a Model** | Basics | [Open Notebook](notebooks/01_how_to_train.ipynb) |
| **How to Generate Text** | Generation | [Open Notebook](notebooks/02_how_to_generate.ipynb) |
| **Reformer Architecture** | Architecture | [Open Notebook](notebooks/03_reformer.ipynb) |
| **Encoder-Decoder Basics** | Architecture | [Open Notebook](notebooks/05_encoder_decoder.ipynb) |
| **Warm Starting Encoder-Decoder** | Training | [Open Notebook](notebooks/08_warm_starting_encoder_decoder.ipynb) |
| **Training Decision Transformers** | RL | [Open Notebook](notebooks/101_train-decision-transformers.ipynb) |
| **Fine-Tuning Whisper** | Audio | [Open Notebook](notebooks/111_fine_tune_whisper.ipynb) |
| **Getting Started with Embeddings** | Embeddings | [Open Notebook](notebooks/80_getting_started_with_embeddings.ipynb) |
| **Sentiment Analysis on Twitter** | NLP | [Open Notebook](notebooks/85_sentiment_analysis_twitter.ipynb) |

[**View all basics notebooks →**](notebooks/)

---

### 🔬 Other Advanced Topics

| Notebook | Topic | Link |
|----------|-------|------|
| **Knowledge Graphs + ChatGPT** | RAG | [Open Notebook](Other/Improve_ChatGPT_with_Knowledge_Graphs.ipynb) |
| **Decoding Strategies in LLMs** | Generation | [Open Notebook](Decoding/Decoding_Strategies_in_Large_Language Models.ipynb) |
| **MergeKit** | Model Merging | [Open Notebook](utils/Mergekit.ipynb) |

---

## ⚡ Quick Start

### 1. Launch Jupyter Lab

```bash
# Simple way
jupyter notebook

# Or with the Makefile (if installed production dependencies)
make notebooks
```

Navigate to any notebook above and start learning!

### 2. Run the Production LLM Server

```bash
# Start the FastAPI server with streaming inference
python Deployment/server.py \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --quantization \
    --port 8000 \
    --max_new_tokens 512
```

### 3. Query the Server

```bash
# Using the client
python Deployment/client.py \
    --endpoint http://localhost:8000/query-stream \
    --query "Explain quantum computing in simple terms"

# Or with curl
curl "http://localhost:8000/query-stream/?query=What%20is%20AI?"
```

### 4. Fine-Tune Your First Model

Open any fine-tuning notebook, for example:
- [Llama 2 Fine-Tuning](Finetuning/Fine_tune_Llama_2_in_Google_Colab.ipynb) - Works on Google Colab free tier
- [Mistral 7B SFT](Finetuning/Fine_tune_Mistral_7b_with_SFT.ipynb) - Fast and efficient
- [Gemma Fine-tuning](Finetuning/Gemma_finetuning_notebook.ipynb) - Latest Google model

---

## 📚 Documentation

### Course Curriculum

The course is organized into 7 comprehensive modules:

#### 1. [Introduction to Generative AI](Docs/1-Introduction-GenAI/README.md)
- Fundamentals of Generative AI
- VAEs, GANs, and Transformer architectures
- Responsible AI, ethics, and governance
- Real-world applications

#### 2. [Natural Language Processing](Docs/2-Natural-Language-Processing/README.md)
- Text preprocessing and tokenization
- Word embeddings (Word2Vec, GloVe, FastText)
- RNNs, LSTMs, and GRUs
- Feature extraction techniques

#### 3. [Large Language Models](Docs/3-Large-Language-Models/README.md)
- Transformer architecture deep dive
- Attention mechanisms and text generation
- Pre-training and fine-tuning strategies
- LoRA, QLoRA, and parameter-efficient methods
- DPO and RLHF
- Model evaluation and benchmarking
- Quantization techniques

#### 4. [Retrieval Augmented Generation](Docs/4-Retrieval-Augmented-Generation/README.md)
- Vector databases and embeddings
- Document ingestion and chunking
- RAG pipelines and orchestration
- Evaluation frameworks

#### 5. [Production Deployment](Docs/5-Deployment-in-Production/README.md)
- LLM APIs and inference optimization
- Prompt engineering
- Production serving (FastAPI, vLLM, TGI)
- Monitoring and scaling

#### 6. [Diffusion Models](Docs/6-Difffusion-Models/README.md)
- Introduction to diffusion processes
- Stable Diffusion and ControlNet
- Fine-tuning and guidance techniques

#### 7. [Multimodal LLMs](Docs/7-Multi-Modal-LLMs/Part-1-Multi-Modal-LLM/README.md)
- Vision-Language models
- Image-text alignment
- Creating custom multimodal models
- Multimodal retrieval systems

---

## 🎯 Key Features

### Core Capabilities

- **Large Language Models (LLMs)**
  - Fine-tuning with LoRA, QLoRA, and full parameter training
  - Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and ORPO
  - Support for Llama 2/3, Mistral, Gemma, Phi-3, and custom architectures
  - Pre-training pipelines with DeepSpeed integration

- **Model Optimization & Quantization**
  - 4-bit/8-bit quantization using BitsAndBytes
  - GPTQ, GGUF, AWQ, and ExLlamaV2 support
  - Memory-efficient inference for consumer hardware

- **Deployment Solutions**
  - Production FastAPI server with streaming inference
  - Multi-worker deployment with load balancing
  - Health monitoring and error handling
  - Docker-ready containerization

- **Multimodal AI**
  - Vision-Language models (IDEFICS, BLIP, LLaVA)
  - Image captioning and visual question answering
  - Cross-modal retrieval and search

- **Transformers Ecosystem**
  - 118+ examples covering NLP, vision, audio, and time-series
  - Semantic segmentation, object detection, and classification
  - Protein folding and DNA sequence modeling

### Production Infrastructure

- **Modern Python Packaging**: `pyproject.toml` with `uv` for fast, reliable dependency management
- **Automated Build System**: Comprehensive Makefile with self-documenting targets
- **Testing Framework**: Pytest integration with coverage reporting
- **Code Quality**: Black, Ruff, and MyPy for formatting, linting, and type checking
- **Documentation**: MkDocs-ready structure for professional documentation
- **CI/CD Ready**: Pre-commit hooks and automated quality checks

---

## 🛠️ Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# All pre-commit checks
make pre-commit
```

### Testing

```bash
# Run tests
make test

# Run tests with coverage
make test-cov

# Run all quality checks
make check
```

### Build and Package

```bash
# Clean build artifacts
make clean

# Lock dependencies
make lock-deps

# Update dependencies
make update-deps
```

---

## 📋 Project Structure

```
Generative-AI-Course/
├── src/                      # Main package source
├── Deployment/               # Production FastAPI server
│   ├── server.py            # Streaming inference server
│   ├── client.py            # Client implementation
│   └── requirements.txt     # Deployment dependencies
├── trainer/                  # Training modules
│   ├── sft.py               # Supervised fine-tuning
│   ├── pretraining.py       # Pre-training pipeline
│   └── requirements.txt     # Training dependencies
├── Data-Preparation/        # 17 dataset creation notebooks
├── Finetuning/             # 17 fine-tuning notebooks
├── Quantization/           # 6 quantization notebooks
├── Multimodal/             # 6 multimodal notebooks
├── Transformers/           # 118+ transformer notebooks
├── Tools/                  # 6 utility notebooks
├── Docs/                   # Course documentation
├── tests/                  # Test suite
├── pyproject.toml          # Package configuration
├── Makefile                # Build automation
└── LICENSE                 # Apache 2.0 License
```

---

## 🔧 Configuration

### Environment Setup

Create a `.env` file for API keys (optional):

```bash
# Hugging Face (for private models)
HF_TOKEN=your_huggingface_token

# Weights & Biases (for experiment tracking)
WANDB_API_KEY=your_wandb_key

# OpenAI (for some data preparation notebooks)
OPENAI_API_KEY=your_openai_key
```

### GPU Configuration

Most notebooks auto-detect GPU. For manual configuration:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features or notebooks
- 📚 Documentation improvements
- 🧪 Test coverage
- 💡 Example applications

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run quality checks (`make pre-commit`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

**You can use this commercially** - it's free for commercial and educational use!

---

## 👤 Author

**Ruslan Magana Vsevolodovna**

- Website: [ruslanmv.com](https://ruslanmv.com)
- GitHub: [@ruslanmv](https://github.com/ruslanmv)
- Email: contact@ruslanmv.com

Professional Data Scientist specializing in Generative AI, Large Language Models, and Production ML Systems.

---

## 🙏 Acknowledgments

This project builds upon the excellent work of:

- Hugging Face for the Transformers library
- The TRL team for training utilities
- DeepSpeed and Accelerate teams
- The open-source AI community

---

## 📊 Project Stats

- **177+ Jupyter Notebooks**: Comprehensive hands-on examples
- **18 Python Modules**: Production-ready implementations
- **7 Course Modules**: Complete curriculum
- **1.2GB+ Resources**: Models, datasets, and examples
- **Apache 2.0 Licensed**: Free for commercial use

---

## 🎓 Skills You'll Master

Upon completing this course, you will master:

- ✅ Large Language Model architecture and fine-tuning
- ✅ Production deployment and optimization
- ✅ Quantization and memory efficiency
- ✅ Multimodal AI systems
- ✅ RAG and vector databases
- ✅ Modern ML engineering practices
- ✅ Enterprise-grade code quality

For a detailed breakdown, see [Essential Skills](Docs/skills.md).

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ruslanmv/Generative-AI-Course/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruslanmv/Generative-AI-Course/discussions)
- **Documentation**: Browse the `/Docs` directory

---

## 🗺️ Roadmap

- [ ] Docker containerization
- [ ] Kubernetes deployment examples
- [ ] Additional model architectures (Claude, GPT-4, etc.)
- [ ] More quantization formats
- [ ] Extended multimodal examples
- [ ] Production monitoring dashboards
- [ ] Automated benchmarking suite

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

**Made with ❤️ by [Ruslan Magana](https://ruslanmv.com)**

**Ready to master Generative AI? [Get Started](#-installation) →**

</div>

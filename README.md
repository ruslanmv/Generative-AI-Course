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
| [LLM Inference with Llama CPP Python (Llama 2.13b Chat)](./Deployment/LLM_Inference_with_llama_cpp_python__Llama_2_13b_chat.ipynb) | Inference with CPP Llama |

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

### Transformers Basic

| Notebook | Description |
| --- | --- |
| [How to Train a Model](./notebooks/how_to_train.ipynb) | Learn the basics of training a model with transformers. |
| [How to Generate Text](./notebooks/how_to_generate.ipynb) | Explore text generation techniques using transformers. |
| [Reformer Architecture](./notebooks/reformer.ipynb) | Dive into the Reformer architecture and its applications. |
| [Encoder-Decoder Basics](./notebooks/encoder_decoder.ipynb) | Understand the fundamentals of encoder-decoder models in transformers. |
| [Warm Starting Encoder-Decoder](./notebooks/warm_starting_encoder_decoder.ipynb) | Learn how to warm start encoder-decoder models for better performance. |
| [Training Decision Transformers](./notebooks/train_decision_transformers.ipynb) | Train decision transformers for various tasks. |
| [TF Serving for Deployment](./notebooks/tf_serving.ipynb) | Deploy transformer models using TF Serving. |
| [Fine-Tuning Whisper Models](./notebooks/fine_tune_whisper.ipynb) | Fine-tune Whisper models for speech recognition tasks. |
| [TF Serving for Vision](./notebooks/tf_serving_vision.ipynb) | Deploy transformer models for vision tasks using TF Serving. |
| [Vertex AI Vision](./notebooks/vertex_ai_vision.ipynb) | Explore Vertex AI Vision for computer vision tasks. |
| [Introducing Contrastive Search](./notebooks/introducing_contrastive_search.ipynb) | Learn about contrastive search and its applications. |
| [CLIPSeg Zero-Shot Learning](./notebooks/clipsseg-zero-shot.ipynb) | Implement zero-shot learning with CLIPSeg. |
| [PyTorch XLA](./notebooks/pytorch_xla.ipynb) | Use PyTorch XLA for accelerated training. |
| [Fine-Tuning Wav2Vec2 for English ASR](./notebooks/fine_tune_wav2vec2_for_english_asr.ipynb) | Fine-tune Wav2Vec2 models for English automatic speech recognition. |
| [Fine-Tuning XLSR Wav2Vec2 ASR](./notebooks/fine_tune_xlsr_wav2vec2_asr.ipynb) | Fine-tune XLSR Wav2Vec2 models for automatic speech recognition. |
| [Constrained Beam Search](./notebooks/constrained_beam_search.ipynb) | Implement constrained beam search for sequence generation. |
| [Fine-Tuning SegFormer](./notebooks/fine_tune_segformer.ipynb) | Fine-tune SegFormer models for image segmentation tasks. |
| [FastAI Hub](./notebooks/fastai_hub.ipynb) | Explore the FastAI Hub for transformer-based models. |
| [Getting Started with Embeddings](./notebooks/getting_started_with_embeddings.ipynb) | Learn the basics of embeddings and their applications. |
| [Sentiment Analysis on Twitter](./notebooks/sentiment_analysis_twitter.ipynb) | Perform sentiment analysis on Twitter data using transformers. |
| [TF XLA Generate](./notebooks/tf_xla_generate.ipynb) | Use TF XLA for accelerated generation tasks. |
| [Training Sentence Transformers](./notebooks/training_sentence_transformers.ipynb) | Train sentence transformers for various NLP tasks. |
| [Federated Learning with Flower](./notebooks/fl-with-flower.ipynb) | Implement federated learning with Flower. |
| [GraphML Classification](./notebooks/graphml-classification.ipynb) | Perform graph classification using GraphML and transformers. |
| [Hugging Face INT8 Demo](./notebooks/huggingface_int8_demo.ipynb) | Explore Hugging Face's INTo for efficient inference.|


### Transformers Advanced
| Notebook | Description |
| --- | --- |
| [Annotated Diffusion](./Transformers/annotated_diffusion.ipynb) | Explore annotated diffusion techniques in transformers. |
| [Audio Classification](./Transformers/audio_classification.ipynb) | Classify audio data using transformer-based models. |
| [Autoformer Transformers Are Effective](./Transformers/autoformer-transformers-are-effective.ipynb) | Investigate the effectiveness of autoformer transformers. |
| [Automatic Mask Generation](./Transformers-automatic_mask_generation.ipynb) | Generate masks automatically using transformer-based models. |
| [Benchmark](./Transformers/benchmark.ipynb) | Benchmark various transformer models for performance. |
| [Causal Language Modeling Flax](./Transformers/causal_language_modeling_flax.ipynb) | Implement causal language modeling using Flax and transformers. |
| [Image Captioning BLIP](./Transformers/image_captioning_blip.ipynb) | Generate image captions using BLIP and transformers. |
| [Image Captioning Pix2Struct](./Transformers/image_captioning_pix2struct.ipynb) | Explore image captioning using Pix2Struct and transformers. |
| [Image Classification TF](./Transformers/image_classification-tf.ipynb) | Classify images using TensorFlow and transformers. |
| [Image Classification](./Transformers/image_classification.ipynb) | Classify images using transformer-based models. |
| [Image Classification Albumentations](./Transformers/image_classification_albumentations.ipynb) | Use Albumentations with transformers for image classification. |
| [Image Classification Kornia](./Transformers/image_classification_kornia.ipynb) | Utilize Kornia with transformers for image classification. |
| [Image Similarity](./Transformers/image_similarity.ipynb) | Measure image similarity using transformer-based models. |
| [Language Modeling TF](./Transformers/language_modeling-tf.ipynb) | Implement language modeling using TensorFlow and transformers. |
| [Language Modeling](./Transformers/language_modeling.ipynb) | Explore language modeling using transformer-based models. |
| [Language Modeling From Scratch TF](./Transformers/language_modeling_from_scratch-tf.ipynb) | Build language models from scratch using TensorFlow and transformers. |
| [Language Modeling From Scratch](./Transformers/language_modeling_from_scratch.ipynb) | Implement language modeling from scratch using transformers. |
| [Masked Language Modeling Flax](./Transformers/masked_language_modeling_flax.ipynb) | Explore masked language modeling using Flax and transformers. |
| [Multiple Choice TF](./Transformers/multiple_choice-tf.ipynb) | Implement multiple choice tasks using TensorFlow and transformers. |
| [Multiple Choice](./Transformers/multiple_choice.ipynb) | Explore multiple choice tasks using transformer-based models. |
| [Multivariate Informer](./Transformers/multivariate_informer.ipynb) | Forecast multivariate time series data using informer and transformers. |
| [Multi-Lingual Speech Recognition](./Transformers/multi_lingual_speech_recognition.ipynb) | Recognize speech in multiple languages using transformers. |
| [Nucleotide Transformer DNA Sequence Modeling](./Transformers/nucleotide_transformer_dna_sequence_modelling.ipynb) | Model DNA sequences using nucleotide transformers. |
| [Nucleotide Transformer DNA Sequence Modeling with PEFT](./Transformers/nucleotide_transformer_dna_sequence_modelling_with_peft.ipynb) | Use PEFT with nucleotide transformers for DNA sequence modeling. |
| [ONNX Export](./Transformers/onnx-export.ipynb) | Export transformer models to ONNX format. |
| [Patch TSMixer](./Transformers/patch_tsmixer.ipynb) | Implement patch-based TSMixer using transformers. |
| [Patch TST](./Transformers/patch_tst.ipynb) | Explore patch-based TST using transformers. |
| [Protein Folding](./Transformers/protein_folding.ipynb) | Predict protein structures using transformer-based models. |
| [Protein Language Modeling TF](./Transformers/protein_language_modeling-tf.ipynb) | Implement protein language modeling using TensorFlow and transformers. |
| [Protein Language Modeling](./Transformers/protein_language_modeling.ipynb) | Explore protein language modeling using transformer-based models. |
| [Question Answering TF](./Transformers/question_answering-tf.ipynb) | Implement question answering using TensorFlow and transformers. |
| [Question Answering](./Transformers/question_answering.ipynb) | Explore question answering using transformer-based models. |
| [Question Answering ORT](./Transformers/question_answering_ort.ipynb) | Use ORT with transformers for question answering. |
| [Segment Anything](./Transformers/segment_anything.ipynb) | Segment objects using transformer-based models. |
| [Semantic Segmentation TF](./Transformers/semantic_segmentation-tf.ipynb) | Implement semantic segmentation using TensorFlow and transformers. |
| [Semantic Segmentation](./Transformers/semantic_segmentation.ipynb) | Explore semantic segmentation using transformer-based models. |
| [Speech Recognition](./Transformers/speech_recognition.ipynb) | Recognize speech using transformer-based models. |
| [Summarization TF](./Transformers/summarization-tf.ipynb) | Implement summarization using TensorFlow and transformers. |
| [Summarization](./Transformers/summarization.ipynb) | Explore summarization using transformer-based models. |
| [Summarization ORT](./Transformers/summarization_ort.ipynb) | Use ORT with transformers for summarization. |
| [Text Classification TF](./Transformers/text_classification-tf.ipynb) | Implement text classification using TensorFlow and transformers. |
| [Text Classification](./Transformers/text_classification.ipynb) | Explore text classification using transformer-based models. |
| [Text Classification Flax](./Transformers/text_classification_flax.ipynb) | Use Flax with transformers for text classification. |
| [Text Classification ORT](./Transformers/text_classification_ort.ipynb) | Use ORT with transformers for text classification. |
| [Text Classification Quantization Inc](./Transformers/text_classification_quantization_inc.ipynb) | Implement text classification with quantization using Inc. |
| [Text Classification Quantization ORT](./Transformers/text_classification_quantization_ort.ipynb) | Use ORT with transformers for text classification with quantization. |
| [Time-Series Transformers](./Transformers/time-series-transformers.ipynb) | Explore time-series forecasting using transformer-based models. |
| [Time Series Datasets](./Transformers/time_series_datasets.ipynb) | Load and explore time series datasets using transformers. |
| [Tokenizer Training](./Transformers/tokenizer_training.ipynb) | Train tokenizers using transformer-based models. |
| [Token Classification TF](./Transformers/token_classification-tf.ipynb) | Implement token classification using TensorFlow and transformers. |
| [Token Classification](./Transformers/token_classification.ipynb) | Explore token classification using transformer-based models. |
| [TPU Training TF](./Transformers/tpu_training-tf.ipynb) | Train models using TPUs and TensorFlow with transformers. |
| [Translation TF](./Transformers/translation-tf.ipynb) | Implement machine translation using TensorFlow and transformers. |
| [Translation](./Transformers/translation.ipynb) | Explore machine translation using transformer-based models. |
| [Video Classification](./Transformers/video_classification.ipynb) | Classify videos using transformer-based models. |
| [Zero-Shot Object Detection with OWLViT](./Transformers/zeroshot_object_detection_with_owlvit.inb) | Implement zero-shot object detection using OWLViT and transformers. |
| [Video Classification](./Transformers/video_classification.ipynb) | Classify videos using transformer-based models. |
| [Zero-Shot Object Detection with OWLViT](./Transformers/zeroshot_object_detection_with_owlvit.ipynb) | Implement zero-shot object detection using OWLViT and transformers. |

### Accelerate Examples
| Notebook | Description |
| --- | --- |
| [Simple CV Example](./Transformers//accelerate_examples/simple_cv_example.ipynb) | Explore computer vision tasks using accelerate. |
| [Simple NLP Example](./Transformers//accelerate_examples/simple_nlp_example.ipynb) | Explore natural language processing tasks using accelerate. |

### Idefics
| Notebook | Description |
| --- | --- |
| [Finetune Image Captioning PEFT](./Transformers//idefics/finetune_image_captioning_peft.ipynb) | Finetune image captioning models using PEFT. |


## Skills of Generative AI

In this section, we summarize the list of essential skills needed for mastering Generative AI. For Data Scientists and AI enthusiasts looking to excel in this rapidly evolving field, it is crucial to build a strong foundation in the core concepts and techniques that power generative models. This includes understanding the architecture of large language models (LLMs), mastering the nuances of text and image generation, and being proficient in advanced methods like reinforcement learning and quantization.

The skills listed are designed to provide a comprehensive overview, covering everything from fundamental text processing techniques to the deployment of multi-modal AI systems capable of handling various data types such as text, images, and audio.

For a detailed breakdown of each skill, along with its description and relevance in the context of Generative AI, please refer to the full document below:

[skills](Docs/skills.md)



## Contributing
--------------

I'm thrilled to have you contribute to this repository! If you're interested in adding new content, fixing bugs, or improving the existing materials. Thank you for helping to make this repository a valuable resource for the Generative AI community!

## About the Author
-------------------

I'm Ruslan Magana Vsevolodovna, a professional Data Scientist with a passion for Generative AI. You can learn more about me and my work at [ruslanmv.com](https://ruslanmv.com).

Thank you for visiting this repository, and I hope you find the resources helpful in your Generative AI journey!
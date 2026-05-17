#  GitaLLM: Custom PyTorch Transformer from Scratch

**Architect:** Arvind K N (AKN)  
**Environment:** Google Colab (NVIDIA T4 GPU)  
**Tech Stack:** PyTorch, Python, Regex, Tiktoken (BPE)

##  Project Overview
GitaLLM is a custom-built, autoregressive Small Language Model (SLM) engineered entirely from scratch in PyTorch. Rather than relying on high-level wrappers or pre-trained weights, this project constructs the raw matrix heart of the AI revolution—the Decoder-Only Transformer—to ingest, process, and generate the philosophical English purports of the *Bhagavad-gita As It Is*.

This repository serves as a mechanical "Go-Kart" to understand the deep systems engineering behind Large Language Models, covering everything from custom causal masking to resolving catastrophic overfitting.

---

##  The Architecture (The Mathematical Edge)

The model is built on a modern Pre-LayerNorm Decoder-Only Transformer architecture.

**Hyperparameters:**
* `vocab_size` = 50,257 (OpenAI BPE Tokenizer)
* `d_model` = 384
* `n_heads` = 6 (64 dimensions per head)
* `n_layers` = 4
* `dropout_rate` = 0.2

### 1. Scaled Dot-Product Attention
The core routing engine of the model. Queries Q, Keys K, and Values V are derived via linear transformations. To prevent the model from peeking at future tokens during autoregressive training, a lower-triangular causal mask M is strictly applied.

### 2. The Residual Highway & Pre-LayerNorm
To prevent vanishing gradients during backpropagation, the model utilizes modern Pre-Layer Normalization combined with standard residual additions. 

x_{attn} = x + Dropout(Attention(LayerNorm(x)))
x_{out} = x_{attn} + Dropout(FFN(LayerNorm(x_{attn})))

### 3. Positional Encoding
Since self-attention contains no inherent recurrence or temporal logic, positional context is injected directly into the input embeddings using interleaved sine and cosine functions:

PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})

---

## The Implementation Pipeline
1. **Data Extraction:** Ripped raw text from the *Bhagavad-gita As It Is* PDF using `PyPDF2`.
2. **Aggressive Sanitization:** Utilized Python Regex to systematically assassinate PDF artifacts, page numbers, and repeating copyright footers. Sliced the text to strictly isolate the English "PURPORT" sections.
3. **Tokenizer Upgrade:** Migrated from a naive Character-Level tokenizer (vocab size: 78) to the industry-standard `tiktoken` Byte-Pair Encoding (vocab size: 50,257) to eliminate "Spanglish" hallucinations.
4. **Training Engine:** PyTorch `AdamW` optimizer utilizing cross-entropy loss against shifted target tensors. 

---

##  The "War Stories" (Failures & Fixes)

Building an engine from scratch means watching it explode. Here are the major architectural bugs encountered and how they were defeated:

### 1. The "PPPPPPPP" Anomaly (The Missing Mask)
**The Bug:** During early generation tests, the model outputted infinite repeating characters (e.g., `PPPPPPPP`). 
**The Autopsy:** The causal mask was omitted during the training loop. The model "cheated" by looking at the target tensor ($Y$) in the future, learning a single rule: *copy the next token*. When forced to generate blindly, it panicked and looped.
**The Fix:** Hardcoded a `torch.tril` lower-triangular matrix directly into the `forward` pass of the `GitaLLM` class.

### 2. The "Copyright" Trap & Spanglish
**The Bug:** The model perfectly memorized the string *"Copyright 1998 The Bhaktivedanta Book Trust"* and mashed English syllables with Sanskrit diacritics (e.g., `sa sac-cid-arjyanra-karma`).
**The Autopsy:** The PDF extractor pulled the footer from every page, and the Character-Level tokenizer failed to understand word boundaries, resulting in alien dialects.
**The Fix:** Regex target-strikes against the copyright strings and swapping to the `gpt2` BPE tokenizer.

### 3. Catastrophic Overfitting
**The Bug:** Validation Loss dropped to `6.6` at step 600, but climbed back to `7.13` while Training Loss magically hit `2.02`. The model began reciting the exact text verbatim.
**The Autopsy:** Pointing a 30-Million parameter brain at a small dataset allowed the model to abandon generalization and simply memorize the training data.
**The Fix:** Added severe regularization. Injected `dropout_rate=0.2` across all embeddings and blocks, implemented `weight_decay=1e-1` in the AdamW optimizer, and engaged early stopping at 1000 iterations.

---

## Future Plans: The Enterprise Pivot
This custom model successfully proved the mechanics of tensor mathematics and Transformer architecture. However, SLMs trained from scratch lack the billions of parameters required for dynamic logical reasoning. 

The next evolution of this project abandons raw training and steps into modern Enterprise AI Architecture:
* **Retrieval-Augmented Generation (RAG):** Integrating `LangChain` to chunk the Gita text.
* **Vector Databases:** Storing embeddings in Chroma/Pinecone to allow a pre-trained LLM (like Llama-3) to retrieve flawless, hallucination-free verses and explanations in real-time.

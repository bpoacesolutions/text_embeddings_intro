# Text Embeddings — Introduction

## Overview

This exercise introduces **text embeddings**, one of the most important building blocks in modern AI systems.

Many AI applications need a way to **compare pieces of text mathematically**. Computers cannot directly understand human language, so we convert text into **numerical representations** that capture semantic meaning.

These numerical representations are called **embeddings**.

Text embeddings are widely used in:

* semantic search
* retrieval-augmented generation (RAG)
* recommendation systems
* document similarity
* AI assistants
* clustering and topic discovery

This example demonstrates how to generate embeddings for sentences using a pretrained model.

---

## What Is a Vector?

In mathematics, a **vector** is an ordered list of numbers.

Example:

```
[0.21, -0.04, 0.78]
```

You may already know vectors from physics or linear algebra, where a vector represents a quantity with **magnitude and direction** in space.

In machine learning, vectors are used to represent **data points in high-dimensional space**.

For example:

```
sentence → vector
```

Instead of 2 dimensions like `(x, y)`, modern language models use **hundreds of dimensions**.

In this example, each sentence becomes a vector with **384 dimensions**:

```
"I love pizza"
↓
[0.12, -0.48, 0.91, 0.03, ..., 0.67]  (384 numbers)
```

This vector is called a **sentence embedding**.

---

## What Do the Dimensions Represent?

Each dimension corresponds to a **latent feature** learned by the model during training.

Unlike traditional data features (such as "age" or "height"), these features are **not explicitly labeled**.

Instead, the model learns patterns in language from large text datasets. These patterns may relate to:

* semantic meaning
* grammar
* context
* relationships between words

For example, words or sentences with similar meanings will tend to produce **vectors that are close together in vector space**.

Example sentences:

```
"I love pizza"
"I enjoy eating pizza"
"The sky is blue"
```

After embedding them:

* the first two vectors will be **close together**
* the third will be **farther away**

Distance between vectors is what allows AI systems to measure **semantic similarity**.

---

## Why Embeddings Matter

Once text is converted into vectors, we can apply mathematical operations such as:

* similarity comparison
* clustering
* nearest-neighbor search
* classification
* semantic retrieval

This is the foundation behind many modern AI systems, including:

* vector databases
* semantic search engines
* document retrieval pipelines
* RAG architectures

---

## Model Used

This example uses a pretrained model from
Hugging Face via the library
sentence-transformers.

Model:

```
all-MiniLM-L6-v2
```

Characteristics:

* lightweight
* fast
* commonly used for semantic search tasks
* produces **384-dimension sentence embeddings**

---

## Installation

Install the required dependency:

```
pip install sentence-transformers
```

---

## Running the Example

Execute the script:

```
python main.py
```

Expected output will look similar to:

```
Sentence: I love pizza
Vector length: 384
First 5 values: [0.021, -0.104, ...]
```

Each sentence is converted into a **384-dimension embedding vector**.

---

## References

* Hugging Face — open-source machine learning platform
* sentence-transformers — library for generating sentence embeddings

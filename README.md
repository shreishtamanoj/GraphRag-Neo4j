# 🧠 GraphRAG with LLM-Powered Q&A

A full **Graph Retrieval-Augmented Generation (GraphRAG)** pipeline that:

* 📄 Ingests documents (PDF, DOCX, TXT, MD)
* ✂️ Splits text into chunks
* 🏷 Extracts entities & relationships using LLM
* 🧮 Generates embeddings
* 🗄 Stores structured data in Neo4j (graph + vector index)
* 🔍 Performs semantic search
* 🤖 Generates contextual answers using LLM
* 🌐 Provides an interactive Gradio UI

---

# 🚀 Features

### ✅ Document Ingestion

* Supports `.pdf`, `.docx`, `.doc`, `.txt`, `.md`
* Extracts text using custom document parser

### ✅ Graph Construction

* Chunk-based storage
* Entity extraction via LLM (Groq - LLaMA 3.3 70B)
* Relationship extraction between entities
* Embedding generation (Sentence-Transformers)
* Vector indexing in Neo4j

### ✅ GraphRAG Query Pipeline

1. Semantic search (vector similarity)
2. Graph traversal (entities + relationships)
3. LLM answer generation with citations

### ✅ Database Utilities

* Clear database
* Remove duplicate relationships
* List documents
* Get database statistics
* Delete specific documents safely

---

# 🏗 Architecture

```
Input Document
    ↓
Text Extraction
    ↓
Chunking (LangChain)
    ↓
Entity & Relationship Extraction (Groq LLM)
    ↓
Embedding Generation (all-MiniLM-L6-v2)
    ↓
Store in Neo4j (Nodes + Relationships + Vector Index)
    ↓
GraphRAG Query Pipeline
    ├─ Vector Search
    ├─ Graph Traversal
    └─ LLM Answer Generation
```

---

# 🗄 Data Model

## Nodes

### `Document`

* `id`
* `filename`
* `created_at`

### `Chunk`

* `id`
* `text`
* `embedding` (384-dim vector)
* `chunk_index`

### `Entity`

* `id`
* `name`
* `type` (Person | Organization | Location | Concept)
* `first_seen_chunk_id`
* `last_updated`

---

## Relationships

* `(:Document)-[:HAS_CHUNK]->(:Chunk)`
* `(:Entity)-[:MENTIONED_IN]->(:Chunk)`
* `(:Entity)-[:RELATION {type: ...}]->(:Entity)`

---

# 🧠 LLM Usage

### During Indexing

* Extract entities
* Extract relationships
* Structure graph data

### During Querying

* Read retrieved chunks
* Use entity graph context
* Generate natural answer with citations

Model Used:

* **Groq API – LLaMA 3.3 70B Versatile**

---

# 🔧 Tech Stack

| Component      | Technology                               |
| -------------- | ---------------------------------------- |
| Graph DB       | Neo4j                                    |
| LLM            | Groq (LLaMA 3.3 70B)                     |
| Embeddings     | Sentence-Transformers (all-MiniLM-L6-v2) |
| Text Splitting | LangChain                                |
| UI             | Gradio                                   |
| Backend        | Python                                   |
| PDF Parsing    | PyPDF                                    |
| DOCX Parsing   | python-docx                              |

---

# 📦 Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Shreishta04/GraphRAG-Neo4j-ChatBot.git
cd GraphRAG-Neo4j-ChatBot
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Setup Environment Variables

Create a `.env` file:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Running the Application

```bash
python app.py
```

Gradio UI will launch in your browser.

---

# 📥 How to Use

## Step 1: Build Knowledge Graph

* Upload a document or paste text
* Click **Build Knowledge Graph**
* System will:

  * Extract text
  * Chunk document
  * Extract entities + relationships
  * Generate embeddings
  * Store everything in Neo4j

---

## Step 2: Ask Questions

* Enter question
* Choose number of sources (top_k)
* Get:

  * LLM-generated answer
  * Source chunks
  * Entities
  * Relationships

---

# 🔍 Example Query

**Question:**

```
Who founded Apple?
```

**System Flow:**

* Finds relevant chunks via vector search
* Traverses graph for related entities
* Sends structured context to LLM
* Returns:

```
Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
(Source: apple_history.pdf - Chunk 2)
```

---

# 🗑 Database Utilities

### Clear Entire Database

```python
clear_database()
```

### Delete Specific Document

```python
delete_document("example.pdf")
```

### Remove Duplicate Relationships

```python
remove_duplicate_relationships()
```

### Get Stats

```python
get_database_stats()
```

---

# 📊 Vector Search Implementation

Uses Neo4j Vector Index:

```cypher
CREATE VECTOR INDEX chunk_embeddings
FOR (c:Chunk)
ON c.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

---

# ⚠️ Challenges Faced

* Duplicate relationships during rebuild
* LLM sometimes returning invalid JSON
* OCR errors affecting entity extraction
* Chunk size affecting graph density
* Managing entity deduplication across documents

---

# 📌 Observations

* Raw text alone is insufficient for structured reasoning
* Graph traversal improves contextual answer generation
* Embeddings + Graph together outperform traditional RAG
* Preprocessing decisions significantly affect retrieval quality

---

# 📈 Future Improvements

* Add relationship type validation
* Add graph visualization
* Add hybrid keyword + vector search
* Add authentication layer
* Add document update support
* Deploy to cloud (Neo4j Aura + Hosted API)

---

# 👩‍💻 Author

Built as part of backend AI research and GraphRAG experimentation.

---

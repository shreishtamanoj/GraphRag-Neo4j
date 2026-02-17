import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import gradio as gr
from document_parser import extract_text_from_file
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize clients
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model (runs locally, no API needed)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, 384 dimensions

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)


def create_vector_index():
    """Create vector index in Neo4j for similarity search"""
    with driver.session() as session:
        # Check if index exists
        result = session.run("SHOW INDEXES")
        indexes = [record["name"] for record in result]
        
        if "chunk_embeddings" not in indexes:
            session.run("""
                CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                FOR (c:Chunk)
                ON c.embedding
                OPTIONS {indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }}
            """)
            print("✅ Vector index created")
        else:
            print("ℹ️  Vector index already exists")


def create_document_node(filename):
    """
    Create or get a Document node in Neo4j
    Returns the document ID
    """
    with driver.session() as session:
        # Check if document already exists
        result = session.run("""
            MATCH (d:Document {filename: $filename})
            RETURN d.id AS id
        """, filename=filename)
        
        record = result.single()
        
        if record:
            # Document exists, return its ID
            print(f"📄 Document '{filename}' already exists, using existing node")
            return record["id"]
        else:
            # Create new document
            doc_id = str(uuid.uuid4())
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    filename: $filename,
                    created_at: datetime(),
                    name: $filename
                })
            """, doc_id=doc_id, filename=filename)
            print(f"📄 Created Document node for '{filename}'")
            return doc_id


def extract_entities(chunk_text, chunk_id):
    """Extract entities and relationships using Groq LLM"""
    
    prompt = f"""Extract entities and relationships from this text.
Return ONLY valid JSON with this exact structure:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Organization|Location|Concept"}}
    ],
    "relationships": [
        {{"source": "entity1", "target": "entity2", "type": "relationship_type"}}
    ]
}}

Text: {chunk_text}

JSON:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000
        )
        
        content = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        content = content.strip()
        
        data = json.loads(content)
        
        # Add chunk metadata to all entities and relationships
        for entity in data.get("entities", []):
            entity["chunk_id"] = chunk_id
            entity["chunk"] = chunk_text
        
        for rel in data.get("relationships", []):
            rel["chunk_id"] = chunk_id
            rel["chunk"] = chunk_text
        
        return data
    
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing error: {e}")
        print(f"Raw response: {content[:200]}...")
        return {"entities": [], "relationships": []}
    
    except Exception as e:
        print(f"❌ Extraction error: {e}")
        return {"entities": [], "relationships": []}


def generate_embedding(text):
    """Generate embedding vector for text"""
    embedding = embedding_model.encode(text)
    return embedding.tolist()  # Convert numpy array to list for Neo4j


def insert_graph(entities, relationships, chunk_text, chunk_id, embedding, doc_id, chunk_index):
    """Insert entities, relationships, chunk with embedding, and link to document into Neo4j"""
    
    with driver.session() as session:
        # Create or update Chunk node with embedding
        # Add a display label for better visualization
        chunk_label = f"Chunk {chunk_index}: {chunk_text[:50]}..." if len(chunk_text) > 50 else f"Chunk {chunk_index}: {chunk_text}"
        
        session.run("""
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text,
                c.embedding = $embedding,
                c.name = $label,
                c.chunk_index = $chunk_index
        """, chunk_id=chunk_id, text=chunk_text, embedding=embedding, label=chunk_label, chunk_index=chunk_index)
        
        # Link Chunk to Document via HAS_CHUNK relationship
        session.run("""
            MATCH (d:Document {id: $doc_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (d)-[:HAS_CHUNK {chunk_index: $chunk_index}]->(c)
        """, doc_id=doc_id, chunk_id=chunk_id, chunk_index=chunk_index)
        
        # Insert entities
        for entity in entities:
            entity_id = str(uuid.uuid4())

            # Inside the entity loop
            session.run("""
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET 
                    e.id = $new_id,
                    e.first_seen_chunk_id = $chunk_id,
                    e.chunk = $chunk  // only on creation
                SET e.last_updated = datetime()
            """, name=entity["name"], type=entity["type"], 
            new_id=entity_id, chunk_id=chunk_id, chunk=chunk_text)

            # Always create the MENTIONED_IN relationship
            session.run("""
                MATCH (e:Entity {name: $name, type: $type})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (e)-[:MENTIONED_IN {chunk_index: $chunk_index}]->(c)
            """, name=entity["name"], type=entity["type"], 
            chunk_id=chunk_id, chunk_index=chunk_index)
        
        
        # Insert relationships (MERGE to avoid duplicates)
        for rel in relationships:
            # Check if relationship already exists
            result = session.run("""
                MATCH (source:Entity {name: $source_name})
                MATCH (target:Entity {name: $target_name})
                OPTIONAL MATCH (source)-[r:RELATION {type: $type}]->(target)
                RETURN r.id AS existing_id, r.chunk AS existing_chunk
            """, 
            source_name=rel["source"],
            target_name=rel["target"],
            type=rel["type"]
            )
            
            record = result.single()
            
            if record and record["existing_id"]:
                # Relationship exists - append chunk info
                existing_chunk = record["existing_chunk"] or ""
                new_chunk = f"{existing_chunk}\n---\n{chunk_text}" if existing_chunk else chunk_text
                
                session.run("""
                    MATCH (source:Entity {name: $source_name})
                    MATCH (target:Entity {name: $target_name})
                    MATCH (source)-[r:RELATION {type: $type}]->(target)
                    SET r.chunk = $new_chunk
                """,
                source_name=rel["source"],
                target_name=rel["target"],
                type=rel["type"],
                new_chunk=new_chunk
                )
            else:
                # Create new relationship
                rel_id = str(uuid.uuid4())
                
                session.run("""
                    MATCH (source:Entity {name: $source_name})
                    MATCH (target:Entity {name: $target_name})
                    CREATE (source)-[r:RELATION {
                        id: $id,
                        type: $type,
                        chunk_id: $chunk_id,
                        chunk: $chunk
                    }]->(target)
                """,
                id=rel_id,
                source_name=rel["source"],
                target_name=rel["target"],
                type=rel["type"],
                chunk_id=chunk_id,
                chunk=chunk_text
                )


def process_document(text, filename="pasted_text.txt"):
    """
    Main pipeline: chunk → extract → vectorize → insert
    
    Args:
        text: Document text content
        filename: Name of the source file (for tracking)
    """
    
    print(f"\n{'='*60}")
    print(f"📄 Processing: {filename}")
    print(f"{'='*60}")
    
    print("\n🔧 Creating vector index...")
    create_vector_index()
    
    print(f"\n📝 Creating Document node for '{filename}'...")
    doc_id = create_document_node(filename)
    
    print("\n📄 Splitting document...")
    chunks = text_splitter.split_text(text)
    print(f"✂️  Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        chunk_index = i + 1  # 1-indexed for human readability
        
        print(f"\n📝 Processing chunk {chunk_index}/{len(chunks)}...")
        
        # Extract entities and relationships
        graph_data = extract_entities(chunk, chunk_id)
        print(f"   Entities: {len(graph_data['entities'])}")
        print(f"   Relationships: {len(graph_data['relationships'])}")
        
        # Generate embedding
        print(f"   🧮 Generating embedding...")
        embedding = generate_embedding(chunk)
        
        # Insert into Neo4j
        print(f"   💾 Inserting into Neo4j...")
        insert_graph(
            graph_data["entities"],
            graph_data["relationships"],
            chunk,
            chunk_id,
            embedding,
            doc_id,
            chunk_index
        )
    
    print(f"\n✅ Pipeline complete for '{filename}'!")
    print(f"   Document ID: {doc_id}")
    print(f"   Total chunks: {len(chunks)}")


def remove_duplicate_relationships():
    """
    Remove duplicate relationships that have the same source, target, and type.
    Keeps the first one created and deletes the rest.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (source)-[r:RELATION]->(target)
            WITH source, target, r.type AS rel_type, collect(r) AS rels
            WHERE size(rels) > 1
            WITH source, target, rel_type, rels[1..] AS duplicates
            UNWIND duplicates AS dup
            DELETE dup
            RETURN count(dup) AS deleted_count
        """)
        
        deleted = result.single()["deleted_count"]
        print(f"🗑️  Removed {deleted} duplicate relationships")
        return deleted


def clear_database():
    """Clear all data from Neo4j"""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🗑️  Database cleared")


def delete_document(filename):
    """
    Delete a specific document and all its chunks, entities, and relationships
    """
    with driver.session() as session:
        # First, get chunks belonging to this document
        result = session.run("""
            MATCH (d:Document {filename: $filename})-[:HAS_CHUNK]->(c:Chunk)
            RETURN collect(c.id) AS chunk_ids
        """, filename=filename)
        
        record = result.single()
        if not record or not record["chunk_ids"]:
            print(f"⚠️  No document found with filename: {filename}")
            return
        
        chunk_ids = record["chunk_ids"]
        
        # Delete entities only mentioned in these chunks
        session.run("""
            MATCH (d:Document {filename: $filename})-[:HAS_CHUNK]->(c:Chunk)
            MATCH (e:Entity)-[:MENTIONED_IN]->(c)
            WITH e, count { (e)-[:MENTIONED_IN]->(otherChunk)<-[:HAS_CHUNK]-(otherDoc:Document)
                            WHERE otherDoc <> d } AS outsideMentions
            WHERE outsideMentions = 0
            DETACH DELETE e
        """, filename=filename)
        
        # Delete chunks and document
        result = session.run("""
            MATCH (d:Document {filename: $filename})
            MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            DETACH DELETE c, d
            RETURN count(c) AS deleted_chunks
        """, filename=filename)
        
        deleted_chunks = result.single()["deleted_chunks"]
        print(f"🗑️  Deleted document '{filename}' and {deleted_chunks} chunks")


def get_database_stats():
    """Get counts of nodes and relationships"""
    with driver.session() as session:
        entity_count = session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]
        chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
        doc_count = session.run("MATCH (d:Document) RETURN count(d) AS count").single()["count"]
        rel_count = session.run("MATCH ()-[r:RELATION]->() RETURN count(r) AS count").single()["count"]
        mention_count = session.run("MATCH ()-[r:MENTIONED_IN]->() RETURN count(r) AS count").single()["count"]
        haschunk_count = session.run("MATCH ()-[r:HAS_CHUNK]->() RETURN count(r) AS count").single()["count"]
        
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "entities": entity_count,
            "relationships": rel_count,
            "mentions": mention_count,
            "has_chunk_links": haschunk_count
        }


def list_documents():
    """List all documents in the database with their chunk counts"""
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            RETURN d.filename AS filename, 
                   d.created_at AS created_at,
                   count(c) AS chunk_count
            ORDER BY d.created_at DESC
        """)
        
        documents = []
        for record in result:
            documents.append({
                "filename": record["filename"],
                "created_at": record["created_at"],
                "chunk_count": record["chunk_count"]
            })
        
        return documents


def semantic_search(query, top_k=3):
    """
    Search for relevant chunks using vector similarity
    Returns chunks + their connected entities + source document
    """
    
    # Generate query embedding
    query_embedding = generate_embedding(query)
    
    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('chunk_embeddings', $top_k, $query_vector)
            YIELD node AS chunk, score
            
            // Get the source document
            OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
            
            // Get entities mentioned in this chunk
            OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(chunk)
            
            // Get relationships between these entities
            OPTIONAL MATCH (e)-[r:RELATION]->(other:Entity)
            WHERE (other)-[:MENTIONED_IN]->(chunk)
            
            RETURN 
                chunk.id AS chunk_id,
                chunk.text AS chunk_text,
                chunk.chunk_index AS chunk_index,
                d.filename AS source_document,
                score,
                collect(DISTINCT {name: e.name, type: e.type}) AS entities,
                collect(DISTINCT {
                    source: e.name, 
                    relation: r.type, 
                    target: other.name
                }) AS relationships
            ORDER BY score DESC
        """, top_k=top_k, query_vector=query_embedding)
        
        results = []
        for record in result:
            results.append({
                "chunk_id": record["chunk_id"],
                "chunk_text": record["chunk_text"],
                "chunk_index": record["chunk_index"],
                "source_document": record["source_document"],
                "similarity_score": record["score"],
                "entities": [e for e in record["entities"] if e["name"]],
                "relationships": [r for r in record["relationships"] if r["source"]]
            })
        
        return results


def ask_question(question, top_k=3):
    """
    Full GraphRAG pipeline:
    1. Retrieve relevant chunks via vector search
    2. Get connected entities and relationships
    3. Generate natural language answer using LLM
    """
    
    print(f"\n🔍 Searching for: '{question}'")
    
    # STEP 1: Retrieve context
    results = semantic_search(question, top_k=top_k)
    
    if not results:
        return {
            "answer": "❌ No relevant information found in the knowledge graph.",
            "sources": []
        }
    
    # STEP 2: Build context from retrieved chunks + graph
    context_parts = []
    for i, result in enumerate(results, 1):
        source_info = f"{result['source_document']} (Chunk {result['chunk_index']})" if result['source_document'] else f"Chunk {result['chunk_index']}"
        
        context_part = f"\n--- Source {i}: {source_info} (Similarity: {result['similarity_score']:.3f}) ---\n"
        context_part += f"Text: {result['chunk_text']}\n"
        
        if result['entities']:
            entity_names = [e['name'] for e in result['entities']]
            context_part += f"Entities: {', '.join(entity_names)}\n"
        
        if result['relationships']:
            rel_strs = [f"{r['source']} --[{r['relation']}]--> {r['target']}" 
                       for r in result['relationships']]
            context_part += f"Relationships: {', '.join(rel_strs)}\n"
        
        context_parts.append(context_part)
    
    full_context = "\n".join(context_parts)
    
    # STEP 3: Generate answer using LLM
    print("🤖 Generating answer...")
    
    prompt = f"""You are a helpful assistant that answers questions based on information from a knowledge graph.

Below is context retrieved from the knowledge graph, including text chunks, entities, and their relationships.

{full_context}

Question: {question}

Instructions:
- Answer the question based ONLY on the provided context
- Be concise and direct
- When citing sources, use the document name and chunk number (e.g., "According to report.pdf (Chunk 3)...")
- If you use information from a specific source, mention it
- If the context doesn't contain enough information to answer, say so
- Use the entities and relationships to provide a complete answer

Answer:"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content.strip()
        
        return {
            "answer": answer,
            "sources": results
        }
    
    except Exception as e:
        return {
            "answer": f"❌ Error generating answer: {str(e)}",
            "sources": results
        }
    
# =========================
# GRADIO FILE PROCESSOR
# =========================

def process_file(file):
    if file is None:
        return "❌ No file uploaded."
    try:
        # Prefer orig_name (Gradio ≥ 4.x usually has it)
        filename = getattr(file, 'orig_name', None) or os.path.basename(file.name)
        
        print(f"→ Uploading as: {filename}")
        
        text = extract_text_from_file(file.name)
        if not text.strip():
            return f"⚠️ Empty content in {filename}"
            
        process_document(text, filename=filename)
        return f"✅ Successfully processed **{filename}**"
    except Exception as e:
        return f"❌ Failed: {str(e)}"


import gradio as gr
import os
import time
from pipeline import (
    process_document, 
    ask_question, 
    driver,
    clear_database,
    remove_duplicate_relationships,
    get_database_stats
)
from document_parser import extract_text_from_file, get_file_info

def process_file_upload(file):
    """Process uploaded file and extract text"""
    if file is None:
        return "", "⚠️ Please upload a file"
    
    try:
        # Get file info
        info = get_file_info(file.name)
        
        # Extract text
        text = extract_text_from_file(file.name)
        
        status = f"""✅ File loaded successfully!

📄 File: {info['name']}
📊 Size: {info['size_kb']:.2f} KB
📝 Extracted: {len(text)} characters
📑 Words: ~{len(text.split())} words

The extracted text is now in the text box below.
Click "Build Knowledge Graph" to process it.
"""
        
        return text, status
    
    except Exception as e:
        return "", f"❌ Error processing file: {str(e)}"


def run_pipeline_from_text(text):
    if not text.strip():
        return "⚠️ Please enter some text or upload a file"
   
    try:
        # Use a fallback name when coming from direct text input
        fallback_filename = "pasted_text_" + str(int(time.time())) + ".txt"
        #               or just: "manual_input.txt"
        #               or:     "clipboard.txt"

        process_document(text, filename=fallback_filename)
       
        stats = get_database_stats()
       
        return f"""✅ Knowledge Graph Built Successfully!
                📊 Database Stats:
                • Chunks: {stats['chunks']}
                • Entities: {stats['entities']}
                • Relationships: {stats['relationships']}
                🔍 Now go to the "Ask Questions" tab to query your knowledge graph!
                """
   
    except Exception as e:
        return f"❌ Error: {str(e)}"

def run_pipeline_from_file(file, text):
    if file is not None:
        try:
            # Get a real filename
            filename = getattr(file, 'orig_name', None) or os.path.basename(file.name)
            extracted_text = extract_text_from_file(file.name)
            
            process_document(extracted_text, filename=filename)   # ← pass filename!
            
            stats = get_database_stats()
            return f"""✅ Processed file **{filename}** successfully!
                📊 Stats: Chunks {stats['chunks']}, Entities {stats['entities']}, Rel {stats['relationships']}"""
            
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"
   
    elif text.strip():
        # Same fallback as above
        fallback = "pasted_text_" + str(int(time.time())) + ".txt"
        try:
            process_document(text, filename=fallback)
            stats = get_database_stats()
            return f"""✅ Processed pasted text as **{fallback}**
            📊 Stats: ..."""
        except Exception as e:
            return f"❌ {str(e)}"
   
    else:
        return "⚠️ Provide file or text"


def query_graph(question, top_k, show_sources):
    if not question.strip():
        return "⚠️ Please enter a question", ""
    
    try:
        result = ask_question(question, top_k=int(top_k))
        
        # Format answer
        answer_output = f"💡 Answer:\n\n{result['answer']}"
        
        # Format sources if requested
        sources_output = ""
        if show_sources and result['sources']:
            sources_output = f"\n📚 Sources Used ({len(result['sources'])} chunks):\n\n"
            
            for i, source in enumerate(result['sources'], 1):
                sources_output += f"{'='*60}\n"
                sources_output += f"Source {i} | Similarity: {source['similarity_score']:.3f}\n"
                sources_output += f"{'='*60}\n\n"
                
                sources_output += f"📄 Chunk:\n{source['chunk_text']}\n\n"
                
                if source['entities']:
                    sources_output += f"🏷️ Entities:\n"
                    for entity in source['entities']:
                        sources_output += f"   • {entity['name']} ({entity['type']})\n"
                    sources_output += "\n"
                
                if source['relationships']:
                    sources_output += f"🔗 Relationships:\n"
                    for rel in source['relationships']:
                        sources_output += f"   • {rel['source']} --[{rel['relation']}]--> {rel['target']}\n"
                    sources_output += "\n"
        
        return answer_output, sources_output
    
    except Exception as e:
        return f"❌ Query error: {str(e)}", ""


def clear_db():
    try:
        clear_database()
        return "🗑️ Database cleared - ready for new documents"
    except Exception as e:
        return f"❌ Error clearing database: {str(e)}"


def remove_duplicates():
    try:
        deleted = remove_duplicate_relationships()
        stats = get_database_stats()
        
        return f"""✅ Duplicate Cleanup Complete!

🗑️ Removed {deleted} duplicate relationships

📊 Current Stats:
• Chunks: {stats['chunks']}
• Entities: {stats['entities']}
• Relationships: {stats['relationships']}
"""
    except Exception as e:
        return f"❌ Error removing duplicates: {str(e)}"


# Gradio Interface
with gr.Blocks(title="GraphRAG with LLM", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🧠 GraphRAG with LLM Answer Generation
    
    **Full Pipeline:** Text → Chunk → Extract Entities → Vectorize → Store in Graph → **LLM-Powered Q&A**
    """)
    
    with gr.Tabs():
        
        # Tab 1: Build Knowledge Graph
        with gr.Tab("📥 Build Knowledge Graph"):
            gr.Markdown("""
            ### Step 1: Load your document
            
            **Option A:** Upload a file (PDF, DOCX, TXT, MD)  
            **Option B:** Paste text directly
            
            The system will:
            1. Extract text from your document
            2. Split text into chunks
            3. Extract entities and relationships (using Groq LLM)
            4. Generate embeddings (using Sentence-Transformers)
            5. Store everything in Neo4j graph database
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 📎 Upload Document")
                    file_input = gr.File(
                        label="Upload File",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md"],
                        type="filepath"
                    )
                    load_file_btn = gr.Button("📖 Load File", variant="secondary")
                    file_status = gr.Textbox(label="File Status", lines=6, interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("#### ✏️ Or Paste Text")
                    text_input = gr.Textbox(
                        label="Text Content",
                        placeholder="Text will appear here after loading file, or paste directly...",
                        lines=12
                    )
            
            load_file_btn.click(
                process_file_upload,
                inputs=file_input,
                outputs=[text_input, file_status]
            )
            
            gr.Markdown("---")
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Build Knowledge Graph", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear Database", variant="stop")
                dedup_btn = gr.Button("🧹 Remove Duplicates", variant="secondary")
            
            build_output = gr.Textbox(label="Status", lines=8)
            
            submit_btn.click(
                run_pipeline_from_file, 
                inputs=[file_input, text_input], 
                outputs=build_output
            )
            clear_btn.click(clear_db, outputs=build_output)
            dedup_btn.click(remove_duplicates, outputs=build_output)
            
            gr.Markdown("""
            ---
            **💡 Supported Formats:**
            - PDF (.pdf) - Extracts text from all pages
            - Word (.docx, .doc) - Extracts paragraphs and tables
            - Text (.txt, .md) - Reads plain text
            
            **💡 Tip:** After building the graph, switch to the "Ask Questions" tab to query it!
            """)
        
        
        # Tab 2: Ask Questions (LLM-Powered)
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("""
            ### Step 2: Ask questions about your data
            
            The system will:
            1. Find relevant chunks using **semantic search** (vector similarity)
            2. Retrieve connected **entities and relationships** from the graph
            3. Use an **LLM to generate a natural answer** based on the context
            """)
            
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., Who founded Apple? What did Steve Jobs do? When was the iPhone released?",
                lines=2
            )
            
            with gr.Row():
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of sources to retrieve"
                )
                
                show_sources_checkbox = gr.Checkbox(
                    label="Show source chunks",
                    value=True
                )
            
            query_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
            
            answer_output = gr.Textbox(label="Answer", lines=8)
            sources_output = gr.Textbox(label="Retrieved Sources (Context Used)", lines=15, visible=True)
            
            query_btn.click(
                query_graph,
                inputs=[question_input, top_k_slider, show_sources_checkbox],
                outputs=[answer_output, sources_output]
            )
            
            # Show/hide sources based on checkbox
            show_sources_checkbox.change(
                lambda x: gr.update(visible=x),
                inputs=show_sources_checkbox,
                outputs=sources_output
            )
            
            gr.Markdown("""
            ---
            ### 🎯 How it works:
            
            **Traditional RAG:**
            - Vector search → Return chunks → Done
            
            **GraphRAG (This System):**
            1. **Vector Search**: Find semantically similar chunks
            2. **Graph Traversal**: Get entities and relationships connected to those chunks
            3. **LLM Generation**: Read all context and generate a natural answer
            
            **Result:** More accurate, context-aware answers with graph-based reasoning!
            """)
        
        
        # Tab 3: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## 🏗️ Architecture
            
            ```
            Input Document
                ↓
            Text Chunking (LangChain)
                ↓
            Entity Extraction (Groq Llama 3.3-70b)
                ↓
            Embedding Generation (Sentence-Transformers)
                ↓
            Store in Neo4j (Chunks + Entities + Relationships + Vectors)
                ↓
            ┌─────────────────────────────────┐
            │  GraphRAG Query Pipeline        │
            ├─────────────────────────────────┤
            │ 1. Vector Search (find chunks)  │
            │ 2. Graph Traversal (get context)│
            │ 3. LLM Answer (Groq)            │
            └─────────────────────────────────┘
            ```
            
            ## 🔧 Tech Stack
            
            - **Graph Database**: Neo4j (nodes + edges + vector index)
            - **Entity Extraction**: Groq API (Llama 3.3-70b)
            - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2, local)
            - **Answer Generation**: Groq API (Llama 3.3-70b)
            - **Text Splitting**: LangChain
            - **UI**: Gradio
            
            ## 📊 Data Model
            
            **Nodes:**
            - `Chunk` - Text chunks with 384-dim embeddings
            - `Entity` - Extracted entities (Person, Organization, Location, Concept)
            
            **Relationships:**
            - `MENTIONED_IN` - Links entities to chunks they appear in
            - `RELATION` - Links entities to each other (type varies: FOUNDED, CEO_OF, etc.)
            
            ## 🚀 LLM Usage Points
            
            1. **During Indexing** (Build Graph):
               - LLM extracts entities and relationships from text
            
            2. **During Querying** (Ask Questions):
               - LLM generates natural language answers from retrieved context
            
            ## 📝 Example Query Flow
            
            **Question:** "Who founded Apple?"
            
            **Step 1 - Retrieval:**
            - Query embedding: `[0.23, -0.45, 0.67, ...]`
            - Vector search finds 3 most similar chunks
            - Graph traversal gets entities: `Steve Jobs`, `Steve Wozniak`, `Ronald Wayne`
            - Relationships: `FOUNDED → Apple Inc.`
            
            **Step 2 - Generation:**
            - LLM reads all retrieved context
            - Generates: "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. (Source 1)"
            
            **Result:** Natural answer with citations!
            """)

if __name__ == "__main__":
    demo.launch()
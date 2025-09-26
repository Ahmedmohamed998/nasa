import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from vector_db_builder import VectorDBBuilder
import json
import functools
from concurrent.futures import ThreadPoolExecutor
import threading

# Only set page config when this module is executed directly (standalone)
if __name__ == "__main__":
    st.set_page_config(
        page_title="Exoplanet AI Assistant",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_chatbot_styles():
    if st.session_state.get("_chatbot_css_injected", False):
        return
    st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        animation: fadeIn 0.5s ease-in;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #e0e0e0;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #2a5298;
        animation: slideIn 0.3s ease-out;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #ff6b6b;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #2a5298;
    }
    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
    .database-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .database-info .status-indicator {
        background-color: #4CAF50;
        box-shadow: 0 0 8px rgba(76, 175, 80, 0.5);
    }
    .quick-question-btn {
        width: 100%;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    .quick-question-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #2a5298;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-ready { background-color: #28a745; }
    .status-loading { background-color: #ffc107; }
    .status-error { background-color: #dc3545; }
    .performance-metrics {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .stChatMessage {
        animation: slideIn 0.3s ease-out;
    }
</style>
""", unsafe_allow_html=True)
    st.session_state["_chatbot_css_injected"] = True

class ExoplanetChatbot:
    """Optimized chatbot class with caching and performance improvements"""
    
    def __init__(self, db_path: str = "./chroma_db", collection_name: str = "exoplanet_docs"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.model = None
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_cleanup = time.time()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._initialization_lock = threading.Lock()
        self._is_initialized = False
        
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        if current_time - self._last_cache_cleanup > 60: 
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            self._last_cache_cleanup = current_time

    def _get_cache_key(self, query: str, n_results: int) -> str:
        """Generate cache key for query"""
        return f"search_{hash(query)}_{n_results}"

    def setup_gemini(self, api_key: str) -> bool:
        """Setup Gemini API with error handling"""
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            return True
        except Exception as e:
            st.error(f"Error setting up Gemini: {str(e)}")
            return False
    
    def setup_chromadb(self) -> bool:
        """Setup ChromaDB connection with optimized embedding function"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            
            embedding_options = [
                lambda: embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                ),
                lambda: embedding_functions.DefaultEmbeddingFunction()
            ]
            
            for i, embedding_func in enumerate(embedding_options):
                try:
                    self.embedding_function = embedding_func()
                    return True
                except Exception as embed_error:
                    if i == len(embedding_options) - 1:
                        raise embed_error
                    continue
            
            return False
            
        except Exception as e:
            st.error(f"Error setting up ChromaDB: {str(e)}")
            return False
    
    def load_collection(self) -> bool:
        """Load existing collection"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            return True
        except Exception as e:
            st.error(f"Error loading collection: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks with caching and performance optimization"""
        if not self.collection:
            return []
        
        cache_key = self._get_cache_key(query, n_results)
        self._cleanup_cache()
        
        if cache_key in self._cache:
            return self._cache[cache_key][0]
        
        try:
            start_time = time.time()
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    chunks.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i]
                    })
            
            self._cache[cache_key] = (chunks, time.time())
            
            search_time = time.time() - start_time
            if 'search_times' not in st.session_state:
                st.session_state.search_times = []
            st.session_state.search_times.append(search_time)
            
            return chunks
            
        except Exception as e:
            st.error(f"Error searching vector database: {str(e)}")
            return []
    
    def detect_query_language(self, query: str) -> str:
        """Detect the language of the user query"""
        if not query:
            return "English"
        
        arabic_chars = sum(1 for char in query if '\u0600' <= char <= '\u06FF')
        total_chars = len([char for char in query if char.isalpha()])
        
        if total_chars == 0:
            return "English"
        
        arabic_ratio = arabic_chars / total_chars
        return "Arabic" if arabic_ratio > 0.3 else "English"

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]], language: str = "Auto") -> str:
        """Generate response using Gemini with automatic language detection"""
        if not context_chunks:
            return "I don't have enough information to answer your question. Please make sure the vector database is built properly with your project documents."
        
        if language == "Auto":
            detected_language = self.detect_query_language(query)
        else:
            detected_language = language
        
        context = "\n\n".join([
            f"Source: {chunk['metadata']['source']} (Language: {chunk['metadata']['language']})\nContent: {chunk['content']}" 
            for chunk in context_chunks[:3]
        ])
        
        prompt = f"""You are an intelligent assistant specialized in the "A World Away: Hunting for Exoplanets with AI" project from NASA Space Apps Challenge 2025.

Available context from project documents:
{context}

User question: {query}

Instructions:
- Detect the language of the user's question and respond in the SAME language
- If the question is in Arabic, respond in Arabic
- If the question is in English, respond in English
- Be accurate and detailed in your response
- Use only the provided context for your answer
- If you cannot find sufficient information in the context, state that clearly
- Maintain a professional and helpful tone

Response:"""
        
        try:
            start_time = time.time()
            response = self.model.generate_content(prompt)
            
            gen_time = time.time() - start_time
            if 'generation_times' not in st.session_state:
                st.session_state.generation_times = []
            st.session_state.generation_times.append(gen_time)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}. Please check your API key and try again."

    def initialize_async(self, api_key: str) -> bool:
        """Initialize chatbot components asynchronously for better UX"""
        with self._initialization_lock:
            if self._is_initialized:
                return True
                
            try:
                if not self.setup_gemini(api_key):
                    return False
                
                if not self.setup_chromadb():
                    return False
                
                if not self.load_collection():
                    return False
                
                self._is_initialized = True
                return True
                
            except Exception as e:
                st.error(f"Initialization error: {str(e)}")
                return False

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        metrics = {
            "cache_size": len(self._cache),
            "is_initialized": self._is_initialized
        }
        
        if 'search_times' in st.session_state and st.session_state.search_times:
            metrics["avg_search_time"] = sum(st.session_state.search_times) / len(st.session_state.search_times)
            metrics["total_searches"] = len(st.session_state.search_times)
        
        if 'generation_times' in st.session_state and st.session_state.generation_times:
            metrics["avg_generation_time"] = sum(st.session_state.generation_times) / len(st.session_state.generation_times)
            metrics["total_generations"] = len(st.session_state.generation_times)
        
        return metrics

def get_database_status():
    """Get current database status"""
    builder = VectorDBBuilder()
    return builder.get_database_info()

def get_quick_questions(language: str) -> List[str]:
    """Get predefined quick questions based on language"""
    if language == "Arabic":
        return [
            "ما هو الهدف الرئيسي لهذا المشروع؟",
            "ما هي بيانات بعثات ناسا المستخدمة في هذا المشروع؟",
            "ما هي طريقة العبور لاكتشاف الكواكب الخارجية؟",
            "ما نوع نموذج الذكاء الاصطناعي المستخدم؟",
            "هل توجد واجهة مستخدم لهذا المشروع؟",
            "من هو الجمهور المستهدف للمشروع؟",
            "كيف يساعد هذا المشروع في اكتشاف الكواكب الخارجية؟",
            "ما هي لغات البرمجة المستخدمة؟"
        ]
    else:
        return [
            "What is the main goal of this project?",
            "Which NASA missions' data are used in this project?",
            "What is the transit method for exoplanet detection?",
            "What kind of AI/ML model is being developed?",
            "Is there a user interface for this project?",
            "What are the potential audiences for this project?",
            "How does this project help in exoplanet discovery?",
            "What programming languages or libraries are used?"
        ]

def main():
    inject_chatbot_styles()
    st.markdown("""
    <div class="main-header">
        <h1>🌍 Exoplanet AI Project Assistant</h1>
        <p>NASA Space Apps Challenge 2025 - Interactive Q&A about our project</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ExoplanetChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    
    if 'initialization_start_time' not in st.session_state:
        st.session_state.initialization_start_time = None
    
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = time.time()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key from https://makersuite.google.com/app/apikey"
        )
        
        language = st.selectbox(
            "🌐 Language / اللغة",
            options=["Auto", "English", "Arabic"],
            help="Auto: Detect language automatically | English: Force English | Arabic: Force Arabic"
        )
        
        st.divider()
        
        st.header("💾 Database Status")
        db_info = get_database_status()
        
        if db_info['exists']:
            st.markdown(f"""
                <div class="database-info">
                    <span class="status-indicator status-ready"></span>
                    ✅ Database Ready<br>
                    Collection: {db_info['collection_name']}<br>
                    Documents: {db_info.get('document_count', 0)}<br>
                    Path: {db_info['db_path']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("""
            ❌ **Database Not Found**
            
            Please build the vector database first using:
            ```bash
            python vector_db_builder.py
            ```
            """)
        
        if st.button("🔄 Rebuild Database", help="Rebuild the vector database from documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔍 Finding documents...")
                progress_bar.progress(20)
                
                builder = VectorDBBuilder()
                doc_files = []
                
                # Find document files
                for file in os.listdir("."):
                    if file.endswith('.docx') and not file.startswith('~'):
                        doc_files.append(file)
                
                if doc_files:
                    status_text.text("🏗️ Building database...")
                    progress_bar.progress(50)
                    
                    success = builder.build_database(doc_files, force_rebuild=True)
                    
                    if success:
                        progress_bar.progress(100)
                        status_text.text("✅ Database rebuilt successfully!")
                        st.success("✅ Database rebuilt successfully!")
                        time.sleep(1)  # Brief pause to show success
                        st.rerun()
                    else:
                        status_text.text("❌ Failed to rebuild database")
                        st.error("❌ Failed to rebuild database")
                else:
                    status_text.text("❌ No .docx files found")
                    st.error("No .docx files found in current directory")
            except Exception as e:
                status_text.text(f"❌ Error: {str(e)}")
                st.error(f"Error rebuilding database: {str(e)}")
            finally:
                progress_bar.empty()
                status_text.empty()
        
        st.divider()
        
        st.header("💬 Chat Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        if st.session_state.chat_history:
            total_messages = len(st.session_state.chat_history)
            user_messages = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            st.info(f"Messages: {total_messages} (You: {user_messages})")
            
            metrics = st.session_state.chatbot.get_performance_metrics()
            if metrics.get('avg_search_time') or metrics.get('avg_generation_time'):
                st.markdown("""
                <div class="performance-metrics">
                    <strong>Performance:</strong><br>
                """, unsafe_allow_html=True)
                
                if metrics.get('avg_search_time'):
                    st.markdown(f"🔍 Avg Search: {metrics['avg_search_time']:.2f}s")
                if metrics.get('avg_generation_time'):
                    st.markdown(f"🤖 Avg Generation: {metrics['avg_generation_time']:.2f}s")
                if metrics.get('cache_size'):
                    st.markdown(f"💾 Cache: {metrics['cache_size']} entries")
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    st.header("💬 Chat with the Assistant")
    
    if not api_key:
        st.warning("👆 Please enter your Gemini API key in the sidebar to start chatting.")
        return
    
    if not db_info['exists']:
        st.error("💾 Vector database not found. Please build it first using the sidebar button or command line.")
        return
    
    if not st.session_state.db_initialized:
        if st.session_state.initialization_start_time is None:
            st.session_state.initialization_start_time = time.time()
        
        init_container = st.container()
        with init_container:
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <div class="loading-spinner"></div>
                <p>🔧 Initializing chatbot components...</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔑 Setting up Gemini API...")
                progress_bar.progress(25)
                
                if not st.session_state.chatbot.setup_gemini(api_key):
                    st.error("Failed to setup Gemini API. Please check your API key.")
                    return
                
                status_text.text("💾 Connecting to ChromaDB...")
                progress_bar.progress(50)
                
                if not st.session_state.chatbot.setup_chromadb():
                    st.error("Failed to setup ChromaDB connection.")
                    return
                
                status_text.text("📚 Loading vector database...")
                progress_bar.progress(75)
                
                if not st.session_state.chatbot.load_collection():
                    st.error("Failed to load vector database collection.")
                    return
                
                status_text.text("✅ Initialization complete!")
                progress_bar.progress(100)
                
                st.session_state.db_initialized = True
                
                init_time = time.time() - st.session_state.initialization_start_time
                st.success(f"✅ Chatbot initialized successfully! (Took {init_time:.2f}s)")
                
                time.sleep(1)  
                st.rerun()
                
            except Exception as e:
                st.error(f"Initialization failed: {str(e)}")
                return
            finally:
                progress_bar.empty()
                status_text.empty()
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources used"):
                    for i, source in enumerate(message["sources"]):
                        st.write(f"**{i+1}. {source['source']}** (Relevance: {source['relevance']:.2f})")
                        st.write(f"Language: {source['language']}")
                        st.write(f"Preview: {source['preview']}")
                        if i < len(message["sources"]) - 1:
                            st.divider()
    

    if language == "Auto":
        placeholder = "Ask in any language... / اسأل بأي لغة..."
    elif language == "Arabic":
        placeholder = "ما هو الهدف الرئيسي لهذا المشروع؟"
    else:
        placeholder = "What is the main goal of this project?"
    
    if prompt := st.chat_input(placeholder):
        if st.session_state.db_initialized:
            st.session_state.last_activity = time.time()
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response_container = st.container()
                
                search_placeholder = st.empty()
                search_placeholder.markdown("🔍 Searching for relevant information...")
                
                start_time = time.time()
                context_chunks = st.session_state.chatbot.search_similar_chunks(prompt, n_results=5)
                search_time = time.time() - start_time
                
                if context_chunks:
                    search_placeholder.markdown("🤖 Generating response...")
                    
                    gen_start_time = time.time()
                    response = st.session_state.chatbot.generate_response(
                        prompt, context_chunks, language
                    )
                    gen_time = time.time() - gen_start_time
                    
                    search_placeholder.empty()
                    
                    with response_container:
                        st.write(response)
                        
                        total_time = time.time() - start_time
                        st.markdown(f"""
                        <div class="performance-metrics">
                            ⚡ Response time: {total_time:.2f}s (Search: {search_time:.2f}s, Generation: {gen_time:.2f}s)
                        </div>
                        """, unsafe_allow_html=True)
                    
                    sources_info = []
                    for chunk in context_chunks[:3]:
                        sources_info.append({
                            "source": chunk['metadata']['source'],
                            "language": chunk['metadata']['language'],
                            "relevance": 1 - chunk['distance'],
                            "preview": chunk['content'][:200] + "..."
                        })
                    
                    with st.expander("📚 Sources used in this response"):
                        for i, source in enumerate(sources_info):
                            st.write(f"**{i+1}. {source['source']}** (Relevance: {source['relevance']:.2f})")
                            st.write(f"Language: {source['language']}")
                            st.write(f"Preview: {source['preview']}")
                            if i < len(sources_info) - 1:
                                st.divider()
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources_info,
                        "performance": {
                            "total_time": total_time,
                            "search_time": search_time,
                            "generation_time": gen_time
                        }
                    })
                    
                else:
                    search_placeholder.empty()
                    error_msg = "❌ No relevant context found. The database might be empty or the search failed."
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
        else:
            st.error("Please wait for the chatbot to initialize first.")
    
    with st.sidebar:
        st.header("ℹ️ Project Info")
        
        # Project overview
        st.markdown("""
        **🎯 About This Project:**
        - **Goal**: Automate exoplanet detection using AI/ML
        - **Data**: NASA's Kepler, K2, and TESS missions
        - **Method**: Transit method detection
        - **AI**: Machine learning classification
        - **Interface**: User-friendly web application
        
        **✨ Features:**
        - Automatic language detection (English/Arabic)
        - Real-time Q&A about the project
        - Document-based knowledge retrieval
        - Contextual responses with source citations
        - Performance optimized with caching
        - Responds in the same language as your question
        """)
        
        st.divider()
        
        st.header("⚡ Quick Questions")
        if language == "Auto":
            quick_questions = [
                "What is the main goal of this project?",
                "ما هو الهدف الرئيسي لهذا المشروع؟",
                "Which NASA data is being used?",
                "ما هي بيانات ناسا المستخدمة؟",
                "How does exoplanet detection work?",
                "كيف يعمل اكتشاف الكواكب الخارجية؟"
            ]
        elif language == "Arabic":
            quick_questions = [
                "ما هو الهدف الرئيسي لهذا المشروع؟",
                "ما هي بيانات ناسا المستخدمة؟",
                "كيف يعمل اكتشاف الكواكب الخارجية؟",
                "ما نوع الذكاء الاصطناعي المستخدم؟"
            ]
        else:
            quick_questions = [
                "What is the main goal of this project?",
                "Which NASA data is being used?",
                "How does exoplanet detection work?",
                "What AI/ML techniques are used?"
            ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{hash(question)}", help="Click to ask this question"):
                # Simulate chat input
                st.session_state.chat_history.append({"role": "user", "content": question})
                st.rerun()
        
        st.divider()
        
        st.header("💡 Tips")
        if language == "Auto":
            tips = [
                "Ask in any language - the bot will respond in the same language",
                "اسأل بأي لغة - سيرد البوت بنفس اللغة",
                "Ask specific questions for better answers",
                "اسأل أسئلة محددة للحصول على إجابات أفضل",
                "The assistant uses actual project documents",
                "المساعد يستخدم وثائق المشروع الفعلية"
            ]
        elif language == "Arabic":
            tips = [
                "اسأل أسئلة محددة للحصول على إجابات أفضل",
                "يمكنك السؤال عن أي جانب من جوانب المشروع",
                "المساعد يستخدم وثائق المشروع الفعلية",
                "جرب أسئلة مختلفة لاستكشاف المشروع",
                "استخدم الأسئلة السريعة أعلاه للبدء"
            ]
        else:
            tips = [
                "Ask specific questions for better answers",
                "You can ask about any aspect of the project",
                "The assistant uses actual project documents",
                "Try different questions to explore the project",
                "Use the quick questions above to get started"
            ]
        
        for tip in tips:
            st.info(f"💡 {tip}")
        
        st.divider()
        st.header("📊 Database Info")
        if db_info['exists']:
            st.success(f"✅ {db_info.get('document_count', 0)} chunks loaded")
        else:
            st.error("❌ Database not available")
        
        if st.session_state.chat_history:
            st.divider()
            st.header("📈 Session Stats")
            total_questions = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
            st.metric("Questions Asked", total_questions)
            
            if 'search_times' in st.session_state and st.session_state.search_times:
                avg_time = sum(st.session_state.search_times) / len(st.session_state.search_times)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")

def add_footer():
    st.markdown("---")
    
    if 'search_times' in st.session_state and st.session_state.search_times:
        total_searches = len(st.session_state.search_times)
        avg_search_time = sum(st.session_state.search_times) / total_searches
        
        st.markdown(f"""
        <div style='text-align: center; color: #666; padding: 0.5rem; font-size: 0.8rem;'>
            <p>⚡ Performance: {total_searches} searches, avg {avg_search_time:.2f}s | 
            💾 Cache: {len(st.session_state.chatbot._cache)} entries</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🌍 <strong>Exoplanet AI Project</strong> | NASA Space Apps Challenge 2025</p>
        <p>Built By Space-Code Team 🚀 using Streamlit, Gemini AI, and ChromaDB | Optimized for Performance</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()
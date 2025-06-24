import streamlit as st
import requests
import json
from typing import Dict, List, Any
import time
import PyPDF2
import docx
import io
from datetime import datetime
import hashlib

# Configuration
st.set_page_config(
    page_title="QueryMind AI - Smart Document Assistant",
    page_icon="🧠",
    layout="wide"
)

class DocumentProcessor:
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading DOCX: {str(e)}"
    
    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            return f"Error reading TXT: {str(e)}"
    
    def process_document(self, uploaded_file) -> Dict[str, Any]:
        """Process uploaded document and extract text"""
        if uploaded_file is None:
            return {"error": "No file uploaded"}
        
        file_type = uploaded_file.type
        file_name = uploaded_file.name
        
        # Create file hash for caching
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        
        text = ""
        if file_type == "application/pdf":
            text = self.extract_text_from_pdf(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = self.extract_text_from_docx(uploaded_file)
        elif file_type == "text/plain":
            text = self.extract_text_from_txt(uploaded_file)
        else:
            return {"error": f"Unsupported file type: {file_type}"}
        
        return {
            "text": text,
            "file_name": file_name,
            "file_type": file_type,
            "file_hash": file_hash,
            "word_count": len(text.split()),
            "char_count": len(text)
        }

class GroqClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[Dict], model: str = "llama3-8b-8192") -> str:
        """Generate response using Groq API"""
        try:
            payload = {
                "messages": messages,
                "model": model,
                "temperature": 0.7,
                "max_tokens": 1024,
                "stream": False
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class TavilyClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search using Tavily API"""
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Search failed: {response.status_code} - {response.text}"}
                
        except Exception as e:
            return {"error": f"Search error: {str(e)}"}

class AgenticAI:
    def __init__(self, groq_client: GroqClient, tavily_client: TavilyClient):
        self.groq = groq_client
        self.tavily = tavily_client
        self.doc_processor = DocumentProcessor()
        self.conversation_history = []
        self.document_context = None
    
    def should_search(self, user_input: str) -> bool:
        """Determine if web search is needed"""
        search_keywords = [
            "current", "recent", "latest", "news", "today", "2024", "2025",
            "what's happening", "update", "price", "stock", "weather",
            "search", "find", "look up", "verify", "fact check", "confirm"
        ]
        return any(keyword in user_input.lower() for keyword in search_keywords)
    
    def analyze_document(self, document_text: str, analysis_type: str = "summary") -> Dict[str, Any]:
        """Analyze document with different analysis types"""
        analysis_prompts = {
            "summary": "Please provide a comprehensive summary of this document, highlighting the main points and key insights.",
            "key_points": "Extract and list the key points, important facts, and main arguments from this document.",
            "questions": "Generate thoughtful questions that could be asked about this document's content.",
            "fact_check": "Identify claims in this document that could benefit from fact-checking or verification with current information.",
            "insights": "Provide analytical insights, patterns, and deeper understanding of the content in this document."
        }
        
        system_message = {
            "role": "system",
            "content": f"""You are an expert document analyst. {analysis_prompts.get(analysis_type, analysis_prompts['summary'])} 
            Be thorough, accurate, and provide actionable insights. Format your response clearly with headings and bullet points where appropriate."""
        }
        
        messages = [
            system_message,
            {"role": "user", "content": f"Document content:\n\n{document_text[:4000]}..."}  # Limit for API
        ]
        
        response = self.groq.generate_response(messages)
        return {"analysis": response, "type": analysis_type}
    
    def chat_with_document(self, user_question: str, document_text: str) -> Dict[str, Any]:
        """Chat about the document with optional web search"""
        needs_search = self.should_search(user_question)
        
        result = {
            "response": "",
            "search_performed": False,
            "search_results": None
        }
        
        context = f"Document content (excerpt):\n{document_text[:3000]}...\n\n"
        
        if needs_search:
            search_results = self.tavily.search(user_question)
            result["search_performed"] = True
            result["search_results"] = search_results
            
            if "error" not in search_results:
                context += "Current web information:\n"
                if "answer" in search_results:
                    context += f"Quick Answer: {search_results['answer']}\n\n"
                
                if "results" in search_results:
                    for i, result_item in enumerate(search_results["results"][:2]):
                        context += f"Source {i+1}: {result_item.get('content', 'N/A')[:300]}...\n\n"
        
        system_message = {
            "role": "system",
            "content": """You are an intelligent document assistant. You have access to a document and can also use web search results when available. 
            Answer questions about the document accurately, and when web search results are provided, use them to provide additional context or verify information. 
            Always distinguish between information from the document and information from web sources."""
        }
        
        messages = [
            system_message,
            {"role": "user", "content": f"{context}\n\nUser Question: {user_question}"}
        ]
        
        response = self.groq.generate_response(messages)
        result["response"] = response
        
        return result
    
    def process_query(self, user_input: str, model: str = "llama3-8b-8192") -> Dict[str, Any]:
        """Process user query with potential web search"""
        result = {
            "response": "",
            "search_performed": False,
            "search_results": None,
            "thinking": ""
        }
        
        # Determine if search is needed
        needs_search = self.should_search(user_input)
        result["thinking"] = f"Analyzing query: {'Search needed' if needs_search else 'No search needed'}"
        
        context = ""
        if needs_search:
            st.info("🔍 Searching the web for current information...")
            search_results = self.tavily.search(user_input)
            result["search_performed"] = True
            result["search_results"] = search_results
            
            if "error" not in search_results:
                # Format search results for context
                context = "Here are the search results:\n\n"
                if "answer" in search_results:
                    context += f"Quick Answer: {search_results['answer']}\n\n"
                
                if "results" in search_results:
                    for i, result_item in enumerate(search_results["results"][:3]):
                        context += f"Source {i+1}: {result_item.get('title', 'N/A')}\n"
                        context += f"URL: {result_item.get('url', 'N/A')}\n"
                        context += f"Content: {result_item.get('content', 'N/A')[:500]}...\n\n"
        
        # Prepare messages for Groq
        system_message = {
            "role": "system",
            "content": """You are a helpful AI assistant. If search results are provided, use them to give accurate and up-to-date information. Always cite your sources when using search results. Be conversational and helpful."""
        }
        
        messages = [system_message]
        
        # Add conversation history (last 6 messages for context)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = user_input
        if context:
            user_message = f"{context}\n\nUser Question: {user_input}"
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        st.info("🧠 Generating response...")
        response = self.groq.generate_response(messages, model)
        result["response"] = response
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return result

def main():
    st.title("🧠 IntelliFlow AI")
    st.markdown("*Smart Document Assistant powered by Groq AI and Tavily Search*")
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📄 Document Analysis", "📊 Analytics"])
    
    # Sidebar for API keys and settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Enter your Groq API key"
        )
        
        tavily_api_key = st.text_input(
            "Tavily API Key", 
            type="password",
            help="Enter your Tavily API key"
        )
        
        model_choice = st.selectbox(
            "Select Model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
            help="Choose the Groq model to use"
        )
        
        st.divider()
        
        if st.button("🗑️ Clear All Data"):
            for key in ["messages", "uploaded_doc", "doc_analysis"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("### 🚀 Features:")
        st.markdown("""
        - **Smart Chat**: Context-aware conversations
        - **Web Search**: Real-time information
        - **Document Analysis**: PDF, DOCX, TXT support
        - **Fact Checking**: Verify document claims
        - **Multi-Modal**: Chat + Documents combined
        """)
    
    # Initialize session state
    for key in ["messages", "uploaded_doc", "doc_analysis"]:
        if key not in st.session_state:
            st.session_state[key] = None if key != "messages" else []
    
    # Check if API keys are provided
    if not groq_api_key or not tavily_api_key:
        st.warning("⚠️ Please enter both Groq and Tavily API keys in the sidebar to continue.")
        st.info("💡 **Getting API Keys:**")
        st.markdown("- **Groq**: Sign up at [console.groq.com](https://console.groq.com)")
        st.markdown("- **Tavily**: Sign up at [tavily.com](https://tavily.com)")
        return
    
    # Initialize clients
    try:
        groq_client = GroqClient(groq_api_key)
        tavily_client = TavilyClient(tavily_api_key)
        agent = AgenticAI(groq_client, tavily_client)
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return
    
    
    # TAB 1: Chat Assistant
    with tab1:
        st.header("💬 Intelligent Chat Assistant")
        
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show search results if available
                if message["role"] == "assistant" and "search_results" in message:
                    if message["search_results"]:
                        with st.expander("🔍 Search Results Used"):
                            search_data = message["search_results"]
                            if "results" in search_data:
                                for i, result in enumerate(search_data["results"][:3]):
                                    st.markdown(f"**{i+1}. {result.get('title', 'N/A')}**")
                                    st.markdown(f"🔗 {result.get('url', 'N/A')}")
                                    st.markdown(f"📝 {result.get('content', 'N/A')[:200]}...")
                                    st.divider()
        
        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = agent.process_query(prompt, model_choice)
                    
                    st.markdown(result["response"])
                    
                    # Show search results if performed
                    if result["search_performed"] and result["search_results"]:
                        if "error" not in result["search_results"]:
                            with st.expander("🔍 Search Results Used"):
                                search_data = result["search_results"]
                                if "results" in search_data:
                                    for i, search_result in enumerate(search_data["results"][:3]):
                                        st.markdown(f"**{i+1}. {search_result.get('title', 'N/A')}**")
                                        st.markdown(f"🔗 {search_result.get('url', 'N/A')}")
                                        st.markdown(f"📝 {search_result.get('content', 'N/A')[:200]}...")
                                        st.divider()
                        else:
                            st.warning(f"Search failed: {result['search_results']['error']}")
            
            # Add assistant response to chat history
            assistant_message = {
                "role": "assistant", 
                "content": result["response"],
                "search_results": result["search_results"] if result["search_performed"] else None
            }
            st.session_state.messages.append(assistant_message)
    
    # TAB 2: Document Analysis
    with tab2:
        st.header("📄 Smart Document Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📤 Upload Document")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['pdf', 'docx', 'txt'],
                help="Upload PDF, DOCX, or TXT files for analysis"
            )
            
            if uploaded_file:
                if st.session_state.uploaded_doc != uploaded_file.name:
                    with st.spinner("Processing document..."):
                        doc_result = agent.doc_processor.process_document(uploaded_file)
                        st.session_state.uploaded_doc = uploaded_file.name
                        st.session_state.doc_content = doc_result
                
                if "error" not in st.session_state.doc_content:
                    st.success("✅ Document processed successfully!")
                    st.info(f"**File:** {st.session_state.doc_content['file_name']}")
                    st.info(f"**Words:** {st.session_state.doc_content['word_count']:,}")
                    st.info(f"**Characters:** {st.session_state.doc_content['char_count']:,}")
                    
                    # Analysis options
                    st.subheader("🔍 Analysis Options")
                    analysis_type = st.selectbox(
                        "Choose analysis type:",
                        ["summary", "key_points", "questions", "fact_check", "insights"],
                        format_func=lambda x: {
                            "summary": "📋 Summary",
                            "key_points": "🎯 Key Points", 
                            "questions": "❓ Generate Questions",
                            "fact_check": "✅ Fact Check Claims",
                            "insights": "💡 Deep Insights"
                        }[x]
                    )
                    
                    if st.button("🚀 Analyze Document", use_container_width=True):
                        with st.spinner("Analyzing document..."):
                            analysis = agent.analyze_document(
                                st.session_state.doc_content['text'], 
                                analysis_type
                            )
                            st.session_state.doc_analysis = analysis
                else:
                    st.error(f"Error: {st.session_state.doc_content['error']}")
        
        with col2:
            if st.session_state.doc_analysis:
                st.subheader(f"📊 Analysis Results - {st.session_state.doc_analysis['type'].title()}")
                st.markdown(st.session_state.doc_analysis['analysis'])
            
            # Document Chat Section
            if 'doc_content' in st.session_state and 'error' not in st.session_state.doc_content:
                st.subheader("💬 Chat with Document")
                st.markdown("*Ask questions about your document with real-time fact-checking*")
                
                doc_question = st.text_input(
                    "Ask about your document:",
                    placeholder="What are the main conclusions? Can you verify this claim?"
                )
                
                if st.button("🔍 Ask Question") and doc_question:
                    with st.spinner("Searching and analyzing..."):
                        doc_chat_result = agent.chat_with_document(
                            doc_question, 
                            st.session_state.doc_content['text']
                        )
                        
                        st.markdown("### 🤖 Response:")
                        st.markdown(doc_chat_result["response"])
                        
                        if doc_chat_result["search_performed"] and doc_chat_result["search_results"]:
                            with st.expander("🌐 Web Sources Used for Verification"):
                                search_data = doc_chat_result["search_results"]
                                if "results" in search_data:
                                    for i, result in enumerate(search_data["results"][:3]):
                                        st.markdown(f"**{i+1}. {result.get('title', 'N/A')}**")
                                        st.markdown(f"🔗 {result.get('url', 'N/A')}")
                                        st.divider()
    
    # TAB 3: Analytics
    with tab3:
        st.header("📊 Usage Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "💬 Chat Messages", 
                len(st.session_state.messages),
                help="Total messages in current session"
            )
        
        with col2:
            doc_status = "✅ Loaded" if 'doc_content' in st.session_state else "❌ None"
            st.metric(
                "📄 Document Status", 
                doc_status,
                help="Current document status"
            )
        
        with col3:
            analysis_count = 1 if st.session_state.doc_analysis else 0
            st.metric(
                "🔍 Analyses Run", 
                analysis_count,
                help="Document analyses performed"
            )
        
        if 'doc_content' in st.session_state and 'error' not in st.session_state.doc_content:
            st.subheader("📄 Document Overview")
            doc_info = st.session_state.doc_content
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**File Name:** {doc_info['file_name']}")
                st.info(f"**File Type:** {doc_info['file_type']}")
            
            with col2:
                st.info(f"**Word Count:** {doc_info['word_count']:,}")
                st.info(f"**Character Count:** {doc_info['char_count']:,}")
            
            # Document preview
            st.subheader("👀 Document Preview")
            preview_text = doc_info['text'][:1000] + "..." if len(doc_info['text']) > 1000 else doc_info['text']
            st.text_area("First 1000 characters:", preview_text, height=200, disabled=True)
        
        st.subheader("🎯 Session Summary")
        if st.session_state.messages:
            st.write("Recent conversation topics:")
            for i, msg in enumerate(st.session_state.messages[-5:]):
                if msg["role"] == "user":
                    st.write(f"• {msg['content'][:100]}...")
        else:
            st.write("No conversations yet. Start chatting to see analytics!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🧠 IntelliFlow AI - Combining Chat, Search, and Document Intelligence"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
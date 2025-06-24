import streamlit as st
import requests
import json
from typing import Dict, List, Any
import time

# Configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="wide"
)

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
        self.conversation_history = []
    
    def should_search(self, user_input: str) -> bool:
        """Determine if web search is needed"""
        search_keywords = [
            "current", "recent", "latest", "news", "today", "2024", "2025",
            "what's happening", "update", "price", "stock", "weather",
            "search", "find", "look up"
        ]
        return any(keyword in user_input.lower() for keyword in search_keywords)
    
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
    st.title("🤖 Agentic AI Assistant")
    st.markdown("*Powered by Groq AI and Tavily Search*")
    
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
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("### 📖 How it works:")
        st.markdown("""
        1. **Smart Detection**: The AI determines if your question needs web search
        2. **Web Search**: Uses Tavily to find current information
        3. **AI Response**: Groq generates informed responses
        4. **Context Aware**: Maintains conversation history
        """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
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

if __name__ == "__main__":
    main()
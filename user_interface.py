"""
Streamlit AI Model Interface
A customizable Streamlit web interface for Azure AI Foundry models to complete your Text Generation App.

To use this interface with your own model:
1. Update the endpoint and api_key variables in the env file or inside the User Interface
2. Modify the system_message for your use case if you need to.
3. To Run the user interface copy and paste the following command in the terminal then press enter:
streamlit run user_interface.py

4. (Optional) Ctrl + Left Click on the localhost URL shown in your terminal to open it the interface in your browser.
"""

import streamlit as st
import time
import os 
from datetime import datetime
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AzureAIInterface:
    def __init__(self, api_key=None, endpoint=None, model_name=None, api_version=None):
        # Configuration - can be updated via Streamlit sidebar
        self.api_key = api_key or os.getenv("api_key")
        self.endpoint = endpoint or os.getenv("endpoint")
        self.model_name = model_name or "grok-4-1-fast-reasoning"
        self.api_version = api_version or "2024-05-01-preview"
        
        # Default system message - customizable in UI
        self.default_system_message = ""
        
        # Default model parameters - customizable in UI
        self.temperature = 0.8
        self.top_p = 0.1
        
        # Initialize client if credentials are available
        self.client = None
        if self.api_key and self.endpoint:
            try:
                self.client = ChatCompletionsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key),
                    api_version=self.api_version
                )
            except Exception as e:
                st.error(f"Failed to initialize Azure AI client: {str(e)}")
    
    def update_config(self, api_key, endpoint, model_name, api_version):
        """Update the configuration and reinitialize client"""
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_version = api_version
        
        if self.api_key and self.endpoint:
            try:
                self.client = ChatCompletionsClient(
                    endpoint=self.endpoint,
                    credential=AzureKeyCredential(self.api_key),
                    api_version=self.api_version
                )
                return True, "Configuration updated successfully!"
            except Exception as e:
                return False, f"Failed to update configuration: {str(e)}"
        return False, "API key and endpoint are required"
    
    def generate_response(self, user_message, system_message=None, conversation_history=None, temperature=None, top_p=None):
        """
        Generate a response from the AI model
        
        Args:
            user_message (str): The user's input message
            system_message (str, optional): Custom system message
            conversation_history (list, optional): Previous conversation messages
            temperature (float, optional): Controls randomness in the response, use lower to be more deterministic. (0-1)
            top_p (float, optional): Controls text diversity by selecting the most probable words until a set probability is reached.(0.01-1)
        
        Returns:
            dict: Response containing the AI's message and metadata
        """
        if not self.client:
            return {
                'success': False,
                'error': 'AI client not initialized. Please check your configuration.',
                'timestamp': time.time()
            }
        
        try:
            # Build messages array
            messages = []
            
            # Add system message
            if system_message:
                messages.append(SystemMessage(content=system_message))
            else:
                messages.append(SystemMessage(content=self.default_system_message))
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history:
                    if msg['role'] == 'user':
                        messages.append(UserMessage(content=msg['content']))
                    elif msg['role'] == 'assistant':
                        messages.append(SystemMessage(content=msg['content']))
            
            # Add current user message
            messages.append(UserMessage(content=user_message))
            
            # Build completion parameters - only include provided parameters
            completion_params = {
                'messages': messages,
                'model': self.model_name
            }
            
            # Add parameters only if they are provided (not None)
            if temperature is not None:
                completion_params['temperature'] = temperature
            if top_p is not None:
                completion_params['top_p'] = top_p
            
            # Validation: temperature cannot be 0 and top_p be 0.01 at the same time
            if (temperature is not None and top_p is not None and 
                temperature == 0 and top_p == 0.01):
                completion_params['top_p'] = 1  # Adjust top_p to avoid the conflict
            
            # Get response from model
            response = self.client.complete(**completion_params)
            
            return {
                'success': True,
                'message': response.choices[0].message.content,
                'model': self.model_name,
                'timestamp': time.time()
            }
            
        except HttpResponseError as e:
            return {
                'success': False,
                'error': f"HTTP Error: {str(e)}",
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Error: {str(e)}",
                'timestamp': time.time()
            }

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'ai_interface' not in st.session_state:
        st.session_state.ai_interface = AzureAIInterface()
    
    if 'system_message' not in st.session_state:
        st.session_state.system_message = st.session_state.ai_interface.default_system_message
    
    # Initialize model parameters in session state
    if 'model_temperature' not in st.session_state:
        st.session_state.model_temperature = st.session_state.ai_interface.temperature
    if 'model_top_p' not in st.session_state:
        st.session_state.model_top_p = st.session_state.ai_interface.top_p
    
    # Initialize parameter visibility toggles (all hidden by default)
    if 'show_temperature' not in st.session_state:
        st.session_state.show_temperature = False
    if 'show_top_p' not in st.session_state:
        st.session_state.show_top_p = False
    
    # Initialize system message height
    if 'system_message_height' not in st.session_state:
        st.session_state.system_message_height = 100
    
    # Initialize reset counter for forcing slider updates
    if 'reset_counter' not in st.session_state:
        st.session_state.reset_counter = 0

def render_sidebar():
    """Render the configuration sidebar"""
    st.sidebar.header("🔧 Configuration")
    
    # Model Configuration
    st.sidebar.subheader("Model Settings")
    
    api_key = st.sidebar.text_input(
        "API Key", 
        value=st.session_state.ai_interface.api_key or "",
        type="password",
        help="Your Azure AI API key"
    )
    
    endpoint = st.sidebar.text_input(
        "Endpoint", 
        value=st.session_state.ai_interface.endpoint or "",
        help="Your Azure AI endpoint URL"
    )
    
    model_name = st.sidebar.text_input(
        "Model Name", 
        value=st.session_state.ai_interface.model_name,
        help="The name of the AI model to use"
    )
    
    api_version = st.sidebar.text_input(
        "API Version", 
        value=st.session_state.ai_interface.api_version,
        help="Azure AI API version"
    )
    
    # Update configuration button
    if st.sidebar.button("Update Configuration"):
        success, message = st.session_state.ai_interface.update_config(
            api_key, endpoint, model_name, api_version
        )
        if success:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    
    # System Message Configuration
    st.sidebar.subheader("System Message")
    system_message = st.sidebar.text_area(
        "System Message",
        value=st.session_state.system_message,
        height=st.session_state.system_message_height,
        help="Define the AI's behavior and personality",
        key="system_message_textarea"
    )
    
    if system_message != st.session_state.system_message:
        st.session_state.system_message = system_message
    
    # Model Parameters Configuration with expandable section
    with st.sidebar.expander("Model Parameters", expanded=False):
        # Initialize session state for model parameters if not exists
        if 'model_temperature' not in st.session_state:
            st.session_state.model_temperature = st.session_state.ai_interface.temperature
        if 'model_top_p' not in st.session_state:
            st.session_state.model_top_p = st.session_state.ai_interface.top_p
        
        # Parameter visibility toggles
        st.markdown("**Parameter Selection:**")
        
        show_temperature = st.checkbox(
            "Temperature",
            value=st.session_state.show_temperature,
            help="Show/hide Temperature parameter"
        )
        
        show_top_p = st.checkbox(
            "Top P",
            value=st.session_state.show_top_p,
            help="Show/hide Top P parameter"
        )
        
        # Update session state for visibility toggles
        st.session_state.show_temperature = show_temperature
        st.session_state.show_top_p = show_top_p
        
        st.markdown("---")
        
        # Initialize parameter values
        temperature = st.session_state.model_temperature
        top_p = st.session_state.model_top_p
        
        # Temperature slider (only show if enabled)
        if show_temperature:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.model_temperature,
                step=0.01,
                help="Controls randomness: 0 = deterministic, 1 = very random",
                key=f"temperature_slider_{st.session_state.reset_counter}"
            )
        
        # Top-p slider (only show if enabled)
        if show_top_p:
            top_p = st.slider(
                "Top P",
                min_value=0.01,
                max_value=1.0,
                value=st.session_state.model_top_p,
                step=0.01,
                help="Controls text diversity by selecting the most probable words until a set probability is reached: lower = more focused",
                key=f"top_p_slider_{st.session_state.reset_counter}"
            )
        
        # Validation warning (only show if both parameters are visible and problematic)
        if show_temperature and show_top_p and temperature == 0.0 and top_p == 0.01:
            st.warning("⚠️ Temperature cannot be 0 and Top-p be 0.01 simultaneously. Top P will be adjusted to 1 when temperature is set to 0 (also known as greedy sampling).")
        
        # Update session state with new values (only if sliders are visible)
        if show_temperature:
            st.session_state.model_temperature = temperature
            st.session_state.ai_interface.temperature = temperature
        if show_top_p:
            st.session_state.model_top_p = top_p
            st.session_state.ai_interface.top_p = top_p
        
        # Reset parameters button
        if st.button("🔄 Reset Parameters to Default"):
            # Reset only visible/enabled parameters to defaults
            reset_params = []
            
            if st.session_state.show_temperature:
                st.session_state.model_temperature = 0.8
                st.session_state.ai_interface.temperature = 0.8
                reset_params.append("Temperature")
            
            if st.session_state.show_top_p:
                st.session_state.model_top_p = 0.1
                st.session_state.ai_interface.top_p = 0.1
                reset_params.append("Top P")
            
            if reset_params:
                # Increment counter to force slider re-render with new keys
                st.session_state.reset_counter += 1
            
            st.rerun()
    
    # Chat Controls
    st.sidebar.subheader("Chat Controls")
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.conversation_history = []
        st.rerun()
    
    # Export conversation
    if st.session_state.conversation_history:
        if st.sidebar.button("📥 Export Conversation"):
            export_conversation()

def export_conversation():
    """Export conversation history as a downloadable file"""
    if not st.session_state.conversation_history:
        st.sidebar.warning("No conversation to export")
        return
    
    # Create export content
    export_content = f"# Conversation Export\n"
    export_content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_content += f"**Model:** {st.session_state.ai_interface.model_name}\n"
    export_content += f"**System Message:** {st.session_state.system_message}\n\n"
    export_content += "---\n\n"
    
    for i, msg in enumerate(st.session_state.conversation_history, 1):
        role = "**User**" if msg['role'] == 'user' else "**Assistant**"
        export_content += f"## Message {i} - {role}\n\n{msg['content']}\n\n"
    
    # Create download button
    st.sidebar.download_button(
        label="Download Conversation",
        data=export_content,
        file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )

def render_chat_interface():
    """Render the main chat interface"""
    st.header("🤖 Text Generation App Interface")
    st.subheader("Powered by Azure AI Foundry")
    
    # Display conversation history
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.conversation_history:
            st.info("👋 Welcome! Type a message below to start the conversation.")
        else:
            for msg in st.session_state.conversation_history:
                if msg['role'] == 'user':
                    with st.chat_message("user", avatar="🙋"):
                        st.write(msg['content'])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(msg['content'])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Display user message immediately
        with st.chat_message("user", avatar="🙋"):
            st.write(user_input)
        
        # Generate AI response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                # Only pass parameters that are enabled (visible)
                kwargs = {
                    'user_message': user_input,
                    'system_message': st.session_state.system_message,
                    'conversation_history': st.session_state.conversation_history[:-1]  # Exclude current user message
                }
                
                # Add parameters only if they are enabled
                if st.session_state.show_temperature:
                    kwargs['temperature'] = st.session_state.model_temperature
                if st.session_state.show_top_p:
                    kwargs['top_p'] = st.session_state.model_top_p
                
                response = st.session_state.ai_interface.generate_response(**kwargs)
            
            if response['success']:
                st.write(response['message'])
                # Add AI response to history
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': response['message']
                })
            else:
                error_msg = f"❌ Error: {response['error']}"
                st.error(error_msg)
                # Add error to history
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': error_msg
                })

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Text Generation App - Streamlit Interface",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    .stSpinner {
        text-align: center;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        width: 100%;
    }
    
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Render main chat interface
    render_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Use the configuration tab to adjust your AI model settings and customize the system message. "
        "You can export your conversation history at any time!"
    )

if __name__ == "__main__":
    main()
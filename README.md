# 📄 FastLearner

A modular Retrieval-Augmented Generation (RAG) application that allows you to chat with your PDF documents using your favorite LLM - whether it's a cloud API or self-hosted models like Ollama.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 💬 **Interactive Chat Interface** - Natural conversation with your PDF documents
- 🔄 **Streaming Responses** - Real-time token-by-token response generation
- 📐 **LaTeX Support** - Render mathematical equations and formulas
- 🔌 **Modular Design** - Easy integration with any LLM service
- 🏠 **Local & Cloud** - Support for both self-hosted (Ollama) and API-based models (Mistral)
- 📤 **Simple Upload** - Drag-and-drop PDF file upload
- 💾 **Session Management** - Maintain conversation history and context

## 🚀 Quick Start

### Prerequisites

- Python 3.10.15 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
   git clone https://github.com/einfachsim/FastLearner
   cd FastLearner
```

2. **Create a virtual environment**
```bash
   python -m venv venv
```

3. **Activate the virtual environment**
```bash
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows (Command Prompt)
   venv\Scripts\activate.bat
   
   # On Windows (PowerShell)
   venv\Scripts\Activate.ps1
```

4. **Install dependencies**
```bash
   pip install -r requirements.txt
```

5. **Set up environment variables** (if using API-based models)
```bash
   # Create a .env file
   touch .env
   
   # Add your API key (example for Mistral)
   echo "MISTRAL_API_KEY=your_api_key_here" >> .env
```

6. **Run the application**
```bash
   streamlit run app.py
```

7. **Open your browser**
   
   The app will automatically open at `http://localhost:5000`

## 🛠️ Configuration

### Supported LLM Providers

Currently supported:
- **Mistral API** - Cloud-based API service
- **Ollama** - Self-hosted local models

### Using Different LLM Providers

The application is designed to be highly modular. To use a different LLM provider:

1. Open `rag.py`
2. Examine the existing implementations (`MistralRAGAgentRemote`, `RAGAgent`)
3. Create your own class following the same interface pattern
4. Update `app.py` to use your new implementation

Example structure:
```python
class YourCustomRAGAgent:
    def __init__(self):
        # Initialize your LLM client
        pass
    
    def ingest_file(self, file_path):
        # Process PDF files
        pass
    def search(self, query):
        # Search for a file using your specified metric
    
    def respond(self, query):
        # Generate streaming response
        pass
```

## 📖 Usage

1. **Upload a PDF** - Click the file uploader in the sidebar and select your PDF document
2. **Wait for Processing** - The document will be processed and indexed
3. **Start Chatting** - Type your questions in the chat input at the bottom
4. **View Responses** - Responses stream in real-time with LaTeX support

### Tips

- Upload multiple PDFs to query across multiple documents
- Use the "Clear Conversation" button to start a new chat
- Use the "Reset All" button to clear all uploaded documents and start fresh


## 🤝 Contributing

Contributions are welcome! Here are some ways you can contribute:

- Add support for new LLM providers
- Improve the UI/UX
- Add support for more document types
- Optimize vector storage and retrieval
- Write tests

Please feel free to submit a Pull Request!


---

**Happy chatting with your PDFs! 📚✨**

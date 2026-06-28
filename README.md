README.md
Upload any PDF. Ask anything. Get instant answers powered by a local LLM.
Built during my Generative AI internship at Infosys Springboard.

## What it does
Upload a PDF document through a simple web interface
Extracts text from the document using OCR
Ask questions in plain English — the LLM reads the document and answers
Handles scanned PDFs that aren't natively text-readable

## Tech Stack
LayerToolUIStreamlitLLMOllama (local model)Text extractionOCR integrationLanguagePython

## Features

PDF upload and real-time document parsing
OCR support for scanned/image-based PDFs
Context-aware Q&A — answers grounded in the uploaded document, not general knowledge
Runs locally — no OpenAI API key or cloud dependency needed
Clean Streamlit interface, no frontend coding required


## How to run
bashgit clone https://github.com/yourusername/document-qa-chatbot
cd document-qa-chatbot
pip install -r requirements.txt
ollama pull mistral        # or whichever model you used
streamlit run chatai.py

## Project structure
document-qa-chatbot/
├── chatai.py            # Streamlit UI, OCR pipeline & LLM logic
├── chat_history.json    # Stores conversation history
├── requirements.txt
├── LICENSE
└── README.md

Also two things worth noting for your repo:
chat_history.json — add this to your .gitignore if it contains any real conversation data from testing. You don't want user input stored in a public repo. Add a blank chat_history.json with just [] as a placeholder instead.
requirements.txt — if you haven't created one yet, run this in your project folder:
bashpip freeze > requirements.txt
That way anyone cloning the repo can install dependencies in one command.

## Learnings

Integrating OCR pipelines with LLM context windows
Prompt engineering for document-grounded responses
Building production-ready AI apps with logging and error handling
Local LLM deployment using Ollama


## Built by

Sanjithavarshini M K — Generative AI Intern, Infosys Springboard (Oct–Dec 2025)

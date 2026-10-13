ContextIQ is a knowledge-based AI chatbot project designed to answer questions using only a user-provided collection of documents, rather than relying on unrestricted external information.

The supplied implementation is a Google Colab PDF question-answering prototype. It extracts text from a PDF, prepares overlapping passages, converts those passages into semantic embeddings, retrieves relevant context for a question, and uses an extractive transformer model to select an answer.

The broader project goal is to build a complete application with a chat frontend, backend API, knowledge-base management, reliable out-of-scope handling, and optional user/admin capabilities.

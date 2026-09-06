# AI Gmail Intelligence Assistant

Welcome to the AI Gmail Intelligence Assistant project! This tool integrates with your Gmail to provide semantic search, smart filtering, and intelligent insights into your inbox using Retrieval-Augmented Generation (RAG).

## Project Structure

- `frontend/`: React application built with Vite and styled with Tailwind CSS for a responsive user interface.
- `backend/`: FastAPI application handling Gmail API authentication, email processing, and RAG pipelines using ChromaDB.

## Features (In Progress)
- Secure Gmail OAuth integration
- Intelligent semantic search across your emails
- Automated categorization and insights

## Setup Instructions

### Prerequisites
- Node.js installed
- Python 3.9+ installed
- Google Cloud Console project with Gmail API enabled (download `credentials.json` to the backend root).

### Backend
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and configure your API keys (e.g., OpenAI or other LLM keys).
4. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install the required npm packages:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

## License
This project is for personal use and development.

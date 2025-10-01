***refined with copilot***

# ⚡ Choose Your Own Adventure
Inspired by interactive fiction like AI Dungeon, this application lets you become the protagonist in your own personalized adventure story. Simply enter any theme - from haunted mansions to space exploration - and watch as AI generates a unique branching narrative with multiple paths and endings. The app features an interactive visual map showing your journey through the story, allowing you to explore different decision paths, with a clean black and white interface that works across devices. Each story node presents concise text (around 40 words) with multiple meaningful choices, creating a dynamic storytelling experience where you control the narrative.

## How to Set Up

### Prerequisites

You'll need these installed on your system:
- Python 3.8 or higher
- Node.js 16 or higher  
- Ollama (for AI story generation)
- uv package manager

### Backend Setup

1. **Clone and navigate to the project**
   ```bash
   git clone https://github.com/paudelsamir/Choose-Your-Own-Adventure.git
   cd Choose-Your-Own-Adventure/backend
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up Ollama**
   ```bash
   # Install Ollama (visit https://ollama.ai for instructions)
   ollama serve
   ollama pull llama3.2
   ```

4. **Configure environment**
   ```bash
   # The .env file is already configured, but you can modify if needed
   # Default settings work for most setups
   ```

5. **Start the backend server**
   ```bash
   uv run python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd ../frontend
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Open your browser**
   - Visit `http://localhost:5173`
   - Backend API runs on `http://localhost:8000`

## How to Use

1. **Enter a theme** - Type anything like "haunted mansion", "space exploration", or "medieval quest"
2. **Wait for generation** - The AI creates your personalized story (usually takes 10-30 seconds)
3. **Make choices** - Read the story snippet and choose from 4 options
4. **Explore the map** - Use the visual graph on the right to see your journey
5. **Try different paths** - Click on previous nodes to explore alternate storylines

## Project Structure

```
Choose-Your-Own-Adventure/
├── backend/                 # FastAPI server
│   ├── core/               # Story generation logic
│   ├── models/             # Database models
│   ├── routers/            # API endpoints
│   └── main.py             # Server entry point
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # React components
│   │   └── App.jsx         # Main app component
│   └── package.json        # Node dependencies
└── README.md              # This file
```

## Technology Stack

**Backend**
- FastAPI for the REST API
- SQLite for data storage
- Ollama + Llama 3.2 for AI story generation
- Pydantic for data validation

**Frontend**  
- React with Vite for fast development
- vis-network for interactive graph visualization
- Modern CSS with flexbox and grid
- Responsive design principles

[TROUBLESHOOT](TROUBLESHOOTING.md)
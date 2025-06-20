# Quick Start Guide - Document Search Web Frontend

## 🚀 Start Everything at Once (Recommended)

The easiest way to get started is using the demo launcher:

```bash
# From the .ben directory
uv run start_demo.py
```

This will:
- ✅ Automatically find the best model checkpoint
- ✅ Start the search API on port 8000
- ✅ Start the web frontend on port 8080
- ✅ Open your browser automatically

## 🔧 Manual Setup (Advanced)

If you prefer to start services separately:

### 1. Start the Search API
```bash
cd .ben/vector_db
uv run search_api.py --checkpoint ../checkpoints/best_model.pt --config redis_config.json
```

### 2. Start the Web Frontend
```bash
cd .ben/web
uv run server.py
```

## 📱 Using the Web Interface

1. **Search**: Type your query and press Enter or click Search
2. **Configure**: Choose number of results (5-50) and text inclusion
3. **Browse**: Click document IDs to copy, expand/collapse content
4. **Monitor**: Check system status and stats in real-time

## 🔍 Example Searches

Try these sample queries:
- `machine learning algorithms`
- `neural network training`
- `data preprocessing techniques`
- `model evaluation metrics`

## 🌐 URLs

- **Web Frontend**: http://localhost:8080
- **Search API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## ⚡ Quick Tips

- **Keyboard Shortcuts**: 
  - `Enter` to search
  - `Ctrl+/` to focus search box
  - `Escape` to close modals

- **Performance**: 
  - Disable "Include full text" for faster searches
  - Use 10-20 results for optimal performance

- **Troubleshooting**:
  - Check that Redis is running
  - Verify the search API is healthy at `/health`
  - Look at browser console for errors

## 🛑 Stopping Services

- **Demo Launcher**: Press `Ctrl+C` to stop all services
- **Manual**: Stop each terminal with `Ctrl+C`

---

**Need help?** Check the full [README.md](README.md) for detailed documentation. 
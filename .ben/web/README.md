# Document Search Web Frontend

A modern, responsive web interface for the Document Search API. Built with vanilla HTML, CSS, and JavaScript for fast performance and easy deployment.

## Features

### 🔍 **Smart Search Interface**
- Real-time search with debounced input
- Configurable result count (5, 10, 20, 50)
- Option to include/exclude full document text
- Keyboard shortcuts (Enter to search, Ctrl+/ to focus)

### 📊 **Rich Results Display**
- Similarity scores with visual indicators
- Expandable/collapsible document content
- Click-to-copy document IDs
- Search time performance metrics
- Responsive card-based layout

### 🔧 **System Monitoring**
- Real-time health status indicator
- System statistics (Redis/SQLite document counts)
- Memory usage monitoring
- Auto-refresh capabilities

### 🎨 **Modern UI/UX**
- Glass-morphism design with backdrop blur
- Smooth animations and transitions
- Mobile-responsive layout
- Dark/light theme ready
- Progressive Web App features

### ⚡ **Performance Optimized**
- Debounced search to prevent API spam
- Efficient DOM manipulation
- Lazy loading for large documents
- Client-side caching
- Performance monitoring

## Quick Start

### Prerequisites

1. **Search API Running**: The search API must be running on `localhost:8000`
   ```bash
   cd .ben/vector_db
   uv run search_api.py --checkpoint ../checkpoints/best_model.pt --config redis_config.json
   ```

2. **Python Environment**: Python 3.7+ with `uv` package manager

### Start the Web Frontend

```bash
# Navigate to the web directory
cd .ben/web

# Start the web server (automatically opens browser)
uv run server.py

# Or specify custom port/host
uv run server.py --port 9000 --host 0.0.0.0

# Or start without opening browser
uv run server.py --no-browser
```

The web interface will be available at: **http://localhost:8080**

## File Structure

```
.ben/web/
├── index.html          # Main HTML structure
├── styles.css          # Modern CSS styling
├── script.js           # JavaScript functionality
├── server.py           # HTTP server with CORS
└── README.md           # This documentation
```

## Usage Guide

### Basic Search

1. **Enter Query**: Type your search query in the main search box
2. **Configure Options**: 
   - Select number of results (5-50)
   - Toggle "Include full text" for complete documents
3. **Search**: Click "Search" button or press Enter
4. **View Results**: Browse results with similarity scores

### Advanced Features

#### Keyboard Shortcuts
- `Enter`: Perform search
- `Ctrl + /`: Focus search input
- `Escape`: Close error modals

#### Result Interactions
- **Click Document ID**: Copy to clipboard
- **Show More/Less**: Expand/collapse long documents
- **Similarity Scores**: Color-coded percentage indicators

#### System Monitoring
- **Status Indicator**: Green (online) / Red (issues)
- **Stats Refresh**: Click refresh button for updated metrics
- **Health Details**: Hover over status for more info

## API Integration

The frontend communicates with the search API using these endpoints:

```javascript
// Health check
GET http://localhost:8000/health

// System statistics  
GET http://localhost:8000/stats

// Document search
POST http://localhost:8000/search
{
  "query": "search terms",
  "k": 10,
  "include_text": true
}
```

## Configuration

### API Base URL
Edit `script.js` to change the API endpoint:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Change this
```

### Search Behavior
Customize search parameters in `script.js`:

```javascript
const DEFAULT_RESULTS = 10;        // Default result count
const SEARCH_DELAY = 300;          // Debounce delay (ms)
```

### Styling
Modify `styles.css` for custom themes:

```css
:root {
  --primary-color: #667eea;        /* Main theme color */
  --secondary-color: #764ba2;      /* Accent color */
  --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```

## Browser Support

### Supported Browsers
- **Chrome**: 60+ ✅
- **Firefox**: 55+ ✅  
- **Safari**: 12+ ✅
- **Edge**: 79+ ✅

### Required Features
- ES6+ JavaScript support
- CSS Grid and Flexbox
- Fetch API
- CSS backdrop-filter (for glass effect)

## Development

### Local Development

1. **Edit Files**: Modify HTML, CSS, or JavaScript
2. **Auto-Reload**: Browser will auto-refresh on file changes
3. **Debug**: Use browser DevTools for debugging
4. **Test**: Verify functionality with different queries

### Adding Features

#### New Search Options
1. Add HTML form elements in `index.html`
2. Update search parameter handling in `script.js`
3. Style new elements in `styles.css`

#### Custom Styling
1. Modify CSS variables in `styles.css`
2. Add new animations or transitions
3. Update responsive breakpoints

#### API Extensions
1. Add new endpoint calls in `script.js`
2. Handle new response formats
3. Update UI to display new data

## Troubleshooting

### Common Issues

#### "System Issues" Status
- **Cause**: Search API not running or unreachable
- **Solution**: Start the search API on port 8000
- **Check**: Visit http://localhost:8000/health

#### CORS Errors
- **Cause**: Browser blocking cross-origin requests
- **Solution**: Use the provided `server.py` (includes CORS headers)
- **Alternative**: Start browser with `--disable-web-security` (development only)

#### Port Already in Use
- **Error**: `OSError: [Errno 48] Address already in use`
- **Solution**: Use different port: `uv run server.py --port 8081`
- **Check**: `lsof -i :8080` to see what's using the port

#### Search Not Working
1. **Check API**: Verify http://localhost:8000/health returns 200
2. **Check Console**: Look for JavaScript errors in browser DevTools
3. **Check Network**: Monitor Network tab for failed requests
4. **Check CORS**: Ensure proper CORS headers in responses

#### Slow Performance
- **Large Documents**: Disable "Include full text" for faster searches
- **Network Issues**: Check API response times
- **Browser Cache**: Clear cache and reload page

### Debug Mode

Enable detailed logging in browser console:

```javascript
// Add to script.js for debugging
console.log('Search request:', searchParams);
console.log('Search response:', data);
```

## Performance Tips

### Optimal Usage
- **Query Length**: 2-50 characters work best
- **Result Count**: Use 10-20 results for best performance
- **Text Inclusion**: Disable for faster searches when full text not needed

### Browser Optimization
- **Cache**: Enable browser caching for static assets
- **Memory**: Close unused tabs to free memory
- **Network**: Use wired connection for best performance

## Security Considerations

### Development vs Production

#### Development (Current Setup)
- CORS allows all origins (`*`)
- No authentication required
- HTTP (not HTTPS) communication
- Local network access only

#### Production Recommendations
- Restrict CORS to specific domains
- Add authentication/authorization
- Use HTTPS for secure communication
- Implement rate limiting
- Add input validation and sanitization

### Data Privacy
- Search queries are sent to the API server
- No client-side data storage (except browser cache)
- Document content displayed in browser memory only

## Future Enhancements

### Planned Features
- [ ] Search history and saved queries
- [ ] Advanced filters (date, document type, etc.)
- [ ] Bulk document operations
- [ ] Export search results
- [ ] Dark/light theme toggle
- [ ] Offline search capability (PWA)
- [ ] Multi-language support
- [ ] Search analytics dashboard

### Technical Improvements
- [ ] Service Worker for offline functionality
- [ ] IndexedDB for client-side caching
- [ ] WebSocket for real-time updates
- [ ] Lazy loading for large result sets
- [ ] Virtual scrolling for performance
- [ ] Search result highlighting
- [ ] Auto-complete suggestions

## Contributing

### Code Style
- Use 4 spaces for indentation
- Follow semantic HTML structure
- Use CSS custom properties for theming
- Write descriptive variable names
- Add comments for complex logic

### Testing
- Test on multiple browsers
- Verify mobile responsiveness
- Check accessibility features
- Validate HTML/CSS
- Test with various search queries

## License

This web frontend is part of the MLX8-W2-DocumentSearch project and follows the same license terms as the main project.

---

**🔍 Happy Searching!** 

For issues or questions, check the browser console for error messages and ensure the search API is running properly. 
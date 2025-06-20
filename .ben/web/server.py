#!/usr/bin/env python3
"""
Simple HTTP server for the Document Search web frontend.

This serves the static HTML, CSS, and JavaScript files with proper CORS headers
to allow communication with the search API running on localhost:8000.

Usage:
    uv run web/server.py [--port 3000] [--host localhost]
"""

import os
import sys
import argparse
import http.server
import socketserver
from urllib.parse import urlparse
import webbrowser
import threading
import time

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support."""
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests."""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests with better error handling."""
        try:
            return super().do_GET()
        except Exception as e:
            print(f"Error serving {self.path}: {e}")
            self.send_error(500, f"Internal server error: {e}")
    
    def log_message(self, format, *args):
        """Custom logging format."""
        print(f"[{self.log_date_time_string()}] {format % args}")

def open_browser(url, delay=1.5):
    """Open browser after a short delay."""
    time.sleep(delay)
    print(f"🌐 Opening browser: {url}")
    webbrowser.open(url)

def main():
    parser = argparse.ArgumentParser(
        description="Serve the Document Search web frontend"
    )
    parser.add_argument(
        '--port', '-p', 
        type=int, 
        default=8888,
        help='Port to serve on (default: 8888)'
    )
    parser.add_argument(
        '--host', 
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--no-browser', 
        action='store_true',
        help='Don\'t automatically open browser'
    )
    
    args = parser.parse_args()
    
    # Change to the web directory
    web_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_dir)
    
    # Check if required files exist
    required_files = ['index.html', 'styles.css', 'script.js']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print(f"   Current directory: {web_dir}")
        sys.exit(1)
    
    # Start server
    try:
        with socketserver.TCPServer((args.host, args.port), CORSHTTPRequestHandler) as httpd:
            url = f"http://{args.host}:{args.port}"
            
            print("🔍 Document Search Web Frontend")
            print("=" * 40)
            print(f"📁 Serving from: {web_dir}")
            print(f"🌐 Server URL: {url}")
            print(f"📡 API URL: http://localhost:8000")
            print("=" * 40)
            print("📋 Available files:")
            for file in os.listdir('.'):
                if os.path.isfile(file):
                    size = os.path.getsize(file)
                    print(f"   {file} ({size:,} bytes)")
            print("=" * 40)
            print("💡 Tips:")
            print("   • Make sure the search API is running on port 8000")
            print("   • Use Ctrl+C to stop the server")
            print("   • Use Ctrl+/ to focus search input")
            print("=" * 40)
            
            # Open browser automatically
            if not args.no_browser:
                browser_thread = threading.Thread(target=open_browser, args=(url,))
                browser_thread.daemon = True
                browser_thread.start()
            
            print(f"🚀 Server started! Press Ctrl+C to stop...")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {args.port} is already in use")
            print(f"   Try a different port: python server.py --port {args.port + 1}")
        else:
            print(f"❌ Server error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
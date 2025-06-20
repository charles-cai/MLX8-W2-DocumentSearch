#!/usr/bin/env python3
"""
Demo launcher for Document Search system.

This script starts both the search API server and web frontend
for easy demonstration and testing.

Usage:
    uv run start_demo.py [--api-port 8000] [--web-port 3000]
"""

import subprocess
import time
import argparse
import signal
import sys
import os
import json
from pathlib import Path

def find_best_checkpoint():
    """Find the best checkpoint automatically."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return None
    
    # Look for checkpoints with "best" in the name
    best_checkpoints = list(checkpoints_dir.glob("*best*.pt"))
    if best_checkpoints:
        return str(best_checkpoints[0])
    
    # Fall back to any .pt file
    all_checkpoints = list(checkpoints_dir.glob("*.pt"))
    if all_checkpoints:
        return str(all_checkpoints[0])
    
    return None

def check_redis_config():
    """Check if Redis config exists."""
    config_path = Path("vector_db/redis_config.json")
    if not config_path.exists():
        return None, "Redis config not found"
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        return str(config_path), None
    except Exception as e:
        return None, f"Invalid Redis config: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Start Document Search demo (API + Web Frontend)"
    )
    parser.add_argument(
        '--api-port', 
        type=int, 
        default=8000,
        help='API server port (default: 8000)'
    )
    parser.add_argument(
        '--web-port', 
        type=int, 
        default=8888,
        help='Web server port (default: 8888)'
    )
    parser.add_argument(
        '--checkpoint',
        help='Path to model checkpoint (auto-detected if not specified)'
    )
    parser.add_argument(
        '--no-web',
        action='store_true',
        help='Start only the API server (no web frontend)'
    )
    
    args = parser.parse_args()
    
    print("🔍 Document Search Demo Launcher")
    print("=" * 50)
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_best_checkpoint()
    
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print("❌ Error: No model checkpoint found")
        print("   Please specify --checkpoint or ensure checkpoints exist in ./checkpoints/")
        sys.exit(1)
    
    print(f"📦 Using checkpoint: {checkpoint_path}")
    
    # Check Redis config
    redis_config, error = check_redis_config()
    if error:
        print(f"❌ Error: {error}")
        print("   Please ensure vector_db/redis_config.json exists and is valid")
        sys.exit(1)
    
    print(f"⚙️  Using Redis config: {redis_config}")
    
    # Start processes
    processes = []
    
    try:
        # Start API server
        print(f"\n🚀 Starting API server on port {args.api_port}...")
        api_cmd = [
            "uv", "run", "vector_db/search_api.py",
            "--checkpoint", checkpoint_path,
            "--config", redis_config,
            "--host", "0.0.0.0",
            "--port", str(args.api_port)
        ]
        
        api_process = subprocess.Popen(
            api_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append(("API Server", api_process))
        
        # Wait for API to start
        print("⏳ Waiting for API server to initialize...")
        time.sleep(3)
        
        if api_process.poll() is not None:
            print("❌ API server failed to start")
            output, _ = api_process.communicate()
            print(f"Error output: {output}")
            sys.exit(1)
        
        print(f"✅ API server running on http://localhost:{args.api_port}")
        
        # Start web frontend (unless disabled)
        if not args.no_web:
            print(f"\n🌐 Starting web frontend on port {args.web_port}...")
            web_cmd = [
                "uv", "run", "web/server.py",
                "--port", str(args.web_port),
                "--host", "0.0.0.0"
            ]
            
            web_process = subprocess.Popen(
                web_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            processes.append(("Web Frontend", web_process))
            
            time.sleep(1)
            
            if web_process.poll() is not None:
                print("❌ Web frontend failed to start")
                output, _ = web_process.communicate()
                print(f"Error output: {output}")
            else:
                print(f"✅ Web frontend running on http://localhost:{args.web_port}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("🎉 Document Search Demo Ready!")
        print("=" * 50)
        print(f"📡 API Server: http://localhost:{args.api_port}")
        print(f"📚 API Docs: http://localhost:{args.api_port}/docs")
        if not args.no_web:
            print(f"🌐 Web Frontend: http://localhost:{args.web_port}")
        print("=" * 50)
        print("💡 Tips:")
        print("   • Try searching for 'machine learning' or 'neural networks'")
        print("   • Check the system stats for Redis/SQLite status")
        print("   • Use Ctrl+C to stop all servers")
        print("=" * 50)
        
        # Keep running until interrupted
        print("\n🔄 Servers running... Press Ctrl+C to stop")
        
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"\n❌ {name} stopped unexpectedly")
                    break
            else:
                continue
            break
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    
    finally:
        # Clean up processes
        for name, process in processes:
            if process.poll() is None:
                print(f"🔄 Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚠️  Force killing {name}...")
                    process.kill()
        
        print("👋 All servers stopped. Goodbye!")

if __name__ == '__main__':
    main() 
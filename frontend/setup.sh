#!/bin/bash

# Setup script for the React frontend

echo "🚀 Setting up LLM Eval Frontend..."

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found. Please run this script from the frontend directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
bun install

# Build for production
echo "🔨 Building for production..."
bun run build

echo "✅ Frontend setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Start the backend server: python api_server.py"
echo "2. Visit http://localhost:8001 to see the dashboard"
echo ""
echo "🔧 For development:"
echo "1. Run 'bun run dev' in this directory"
echo "2. Start the backend server in another terminal"
echo "3. Visit http://localhost:3000 for hot reload development"
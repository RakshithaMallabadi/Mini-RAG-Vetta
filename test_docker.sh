#!/bin/bash
# Quick test script for Docker setup

echo "=== Testing Docker Setup ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi
echo "✅ Docker is installed"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "⚠️  docker-compose not found, but Docker is available"
else
    echo "✅ docker-compose is installed"
fi

# Check if Dockerfile exists
if [ -f Dockerfile ]; then
    echo "✅ Dockerfile exists"
else
    echo "❌ Dockerfile not found"
    exit 1
fi

# Check if docker-compose.yml exists
if [ -f docker-compose.yml ]; then
    echo "✅ docker-compose.yml exists"
else
    echo "❌ docker-compose.yml not found"
    exit 1
fi

echo ""
echo "=== Ready to build ==="
echo "Run: docker-compose up --build"

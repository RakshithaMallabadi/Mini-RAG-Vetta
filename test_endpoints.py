#!/usr/bin/env python3
"""
Test script to check all API endpoints
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, files=None, description=""):
    """Test an endpoint and return the result"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n{'='*70}")
    print(f"Testing: {method} {endpoint}")
    if description:
        print(f"Description: {description}")
    print(f"{'='*70}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, data=data, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"Status Code: {response.status_code}")
        
        try:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        except:
            print(f"Response: {response.text[:200]}")
        
        if response.status_code < 400:
            print("✅ SUCCESS")
            return True
        else:
            print("❌ FAILED")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ FAILED - Cannot connect to server")
        return False
    except requests.exceptions.Timeout:
        print("❌ FAILED - Request timeout")
        return False
    except Exception as e:
        print(f"❌ FAILED - Error: {str(e)}")
        return False

def main():
    print("="*70)
    print("Mini RAG 2 API - Endpoint Testing")
    print("="*70)
    
    results = {}
    
    # 1. Root endpoint
    results["GET /"] = test_endpoint("GET", "/", description="Root endpoint with API info")
    
    # 2. Health check
    results["GET /health"] = test_endpoint("GET", "/health", description="Health check")
    
    # 3. Stats (may fail if vector store not loaded)
    results["GET /stats"] = test_endpoint("GET", "/stats", description="Vector store statistics")
    
    # 4. Process documents
    process_data = {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "embedding_model": "all-MiniLM-L6-v2",
        "index_type": "flat"
    }
    results["POST /process"] = test_endpoint("POST", "/process", data=process_data, description="Process documents")
    
    # 4b. Upload document (test with a simple text file)
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for the upload endpoint.")
            temp_file = f.name
        
        with open(temp_file, 'rb') as file:
            upload_data = {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": "all-MiniLM-L6-v2",
                "index_type": "flat"
            }
            upload_files = {"file": ("test.txt", file, "text/plain")}
            results["POST /upload"] = test_endpoint("POST", "/upload", data=upload_data, files=upload_files, description="Upload document")
        
        import os
        os.unlink(temp_file)
    except Exception as e:
        print(f"⚠️  Could not test /upload: {e}")
        results["POST /upload"] = False
    
    # 5. Search (requires vector store)
    search_data = {
        "query": "test query",
        "k": 3,
        "mode": "semantic"
    }
    results["POST /search"] = test_endpoint("POST", "/search", data=search_data, description="Semantic search")
    
    # 6. Answer (requires vector store and API key)
    answer_data = {
        "question": "What is this about?",
        "k": 3,
        "mode": "semantic",
        "temperature": 0.7,
        "max_tokens": 200
    }
    results["POST /answer"] = test_endpoint("POST", "/answer", data=answer_data, description="RAG-based Q&A")
    
    # 7. Check docs endpoint
    results["GET /docs"] = test_endpoint("GET", "/docs", description="Interactive API documentation")
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for endpoint, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {endpoint}")
    
    print(f"\nTotal: {passed}/{total} endpoints working")
    
    if passed == total:
        print("\n🎉 All endpoints are working!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} endpoint(s) need attention")
        return 1

if __name__ == "__main__":
    sys.exit(main())


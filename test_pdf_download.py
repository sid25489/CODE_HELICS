#!/usr/bin/env python3
"""
Test script for PDF download functionality
"""

import requests
import json
import os
from datetime import datetime

def test_pdf_download():
    """Test the PDF download API endpoint"""
    
    # Base URL for the API
    base_url = "http://localhost:5000"
    
    # First, let's test a DNA analysis to get results
    print("Step 1: Testing DNA analysis...")
    
    # Sample DNA sequence
    test_sequence = "ATCGATCGATCGTAGCTAGCTAGCTAATCGATCGTTAACCGGTTAACCGGTTAAGCTAGCTAGCTACGGAATTCCGGAATTCCGGA"
    
    # Analyze DNA sequence
    analyze_response = requests.post(
        f"{base_url}/api/analyze-dna",
        headers={'Content-Type': 'application/json'},
        json={'sequence': test_sequence}
    )
    
    if analyze_response.status_code != 200:
        print(f"❌ DNA analysis failed: {analyze_response.status_code}")
        print(f"Response: {analyze_response.text}")
        return False
    
    analysis_data = analyze_response.json()
    print(f"✅ DNA analysis successful!")
    print(f"   Sequence length: {analysis_data['sequence_length']}")
    print(f"   Number of results: {len(analysis_data['results'])}")
    
    # Step 2: Test PDF download
    print("\nStep 2: Testing PDF download...")
    
    pdf_response = requests.post(
        f"{base_url}/api/download-pdf",
        headers={'Content-Type': 'application/json'},
        json=analysis_data
    )
    
    if pdf_response.status_code != 200:
        print(f"❌ PDF download failed: {pdf_response.status_code}")
        print(f"Response: {pdf_response.text}")
        return False
    
    # Check if response is a PDF
    content_type = pdf_response.headers.get('content-type', '')
    if 'application/pdf' not in content_type:
        print(f"❌ Response is not a PDF. Content-Type: {content_type}")
        return False
    
    # Save the PDF file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"test_dna_report_{timestamp}.pdf"
    
    with open(filename, 'wb') as f:
        f.write(pdf_response.content)
    
    file_size = os.path.getsize(filename)
    print(f"✅ PDF download successful!")
    print(f"   File saved as: {filename}")
    print(f"   File size: {file_size} bytes")
    
    # Verify PDF content (basic check)
    if file_size < 1000:  # PDF should be at least 1KB
        print(f"⚠️  Warning: PDF file seems too small ({file_size} bytes)")
        return False
    
    # Check PDF header
    with open(filename, 'rb') as f:
        header = f.read(4)
        if header != b'%PDF':
            print(f"❌ File doesn't appear to be a valid PDF")
            return False
    
    print(f"✅ PDF file appears to be valid!")
    return True

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('message', 'OK')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Is it running on localhost:5000?")
        return False

if __name__ == "__main__":
    print("🧬 DNA Analyzer PDF Download Test")
    print("=" * 40)
    
    # Test health check first
    if not test_health_check():
        print("\n❌ Server is not running or not accessible")
        print("Please start the server with: python app_minimal.py")
        exit(1)
    
    print()
    
    # Test PDF download functionality
    if test_pdf_download():
        print("\n🎉 All PDF download tests passed!")
    else:
        print("\n❌ PDF download tests failed!")
        exit(1)

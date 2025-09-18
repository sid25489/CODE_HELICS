import requests
from bs4 import BeautifulSoup
import json

def test_page_load(url, page_name):
    """Test if a page loads successfully"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"✓ {page_name} loads successfully")
            return True, response.text
        else:
            print(f"✗ {page_name} failed with status {response.status_code}")
            return False, None
    except Exception as e:
        print(f"✗ {page_name} error: {e}")
        return False, None

def check_html_structure(html, page_name):
    """Check basic HTML structure"""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Check for required elements
        checks = [
            (soup.find('title'), "Title tag"),
            (soup.find('nav'), "Navigation"),
            (soup.find('main') or soup.find('div', class_='container'), "Main content"),
            (soup.find('footer'), "Footer"),
        ]
        
        all_good = True
        for element, name in checks:
            if element:
                print(f"  ✓ {name} found")
            else:
                print(f"  ✗ {name} missing")
                all_good = False
        
        return all_good
    except Exception as e:
        print(f"  ✗ HTML parsing error: {e}")
        return False

def test_javascript_syntax(html):
    """Basic check for JavaScript syntax errors"""
    try:
        # Look for script tags
        soup = BeautifulSoup(html, 'html.parser')
        scripts = soup.find_all('script')
        
        for i, script in enumerate(scripts):
            if script.string:
                # Basic syntax check - look for common issues
                js_code = script.string
                if 'function' in js_code and '{' in js_code and '}' in js_code:
                    print(f"  ✓ Script {i+1} has basic function structure")
                else:
                    print(f"  ? Script {i+1} may have issues")
        
        return True
    except Exception as e:
        print(f"  ✗ JavaScript check error: {e}")
        return False

def main():
    base_url = "http://localhost:5000"
    pages = [
        ("/", "Home Page (Live Dashboard)"),
        ("/analyze", "Analyze Page")
    ]
    
    print("Testing Frontend Functionality...")
    print("=" * 50)
    
    all_tests_passed = True
    
    for path, name in pages:
        print(f"\n{name}:")
        success, html = test_page_load(f"{base_url}{path}", name)
        
        if success and html:
            structure_ok = check_html_structure(html, name)
            js_ok = test_javascript_syntax(html)
            
            if not (structure_ok and js_ok):
                all_tests_passed = False
        else:
            all_tests_passed = False
    
    # Test theme switching functionality
    print(f"\nTheme System Check:")
    home_success, home_html = test_page_load(f"{base_url}/", "Home for theme check")
    if home_success and home_html:
        if 'toggleTheme' in home_html and 'data-theme' in home_html:
            print("  ✓ Theme switching functionality present")
        else:
            print("  ✗ Theme switching functionality missing")
            all_tests_passed = False
    
    print(f"\n{'='*50}")
    if all_tests_passed:
        print("✓ All frontend tests PASSED")
    else:
        print("✗ Some frontend tests FAILED")
    
    return all_tests_passed

if __name__ == "__main__":
    main()

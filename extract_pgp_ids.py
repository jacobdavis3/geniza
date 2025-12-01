#!/usr/bin/env python3
"""
Extract PGP IDs from Princeton Geniza search results.
"""

import requests
from bs4 import BeautifulSoup
import re
import time
import json

def extract_pgp_ids_from_page(url, page_num=1):
    """Extract PGP IDs from a single page of search results."""
    # Add page parameter if not first page
    if page_num > 1:
        # Use page parameter as shown in the URL structure
        if '&page=' in url:
            # Replace existing page parameter
            page_url = re.sub(r'&page=\d+', f'&page={page_num}', url)
        elif '?' in url:
            page_url = f"{url}&page={page_num}"
        else:
            page_url = f"{url}?page={page_num}"
    else:
        # Remove page parameter from first page if present
        page_url = re.sub(r'&page=\d+', '', url)
    
    print(f"Fetching page {page_num}...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pgp_ids = []
        
        # Primary method: Extract from href attributes of view-link elements
        # The PGP ID is in URLs like /en/documents/699/ or /documents/609/
        view_links = soup.find_all('a', class_='view-link', href=True)
        if not view_links:
            # Try finding any links with 'documents' in href
            view_links = soup.find_all('a', href=re.compile(r'/documents/\d+'))
        
        for link in view_links:
            href = link.get('href', '')
            # Extract PGP ID from /en/documents/PGPID/ or /documents/PGPID/
            url_match = re.search(r'/documents/(\d+)', href)
            if url_match:
                pgp_id = url_match.group(1)
                if pgp_id.isdigit():
                    pgp_ids.append(int(pgp_id))
        
        # Fallback method: Look for text containing "PGPID" followed by a number
        if not pgp_ids:
            text_content = soup.get_text()
            pattern = r'PGPID\s*:?\s*(\d+)'
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            pgp_ids.extend([int(m) for m in matches if m.isdigit()])
        
        # Remove duplicates
        pgp_ids = list(set(pgp_ids))
        
        return pgp_ids, soup
        
    except Exception as e:
        print(f"Error fetching page {page_num}: {e}")
        return [], None


def get_total_results(soup):
    """Try to determine total number of results from the HTML."""
    if not soup:
        return 0
    
    # Look for "390 results" text
    text_content = soup.get_text()
    
    # Try to find "X results" pattern
    results_pattern = r'(\d+)\s+results?'
    match = re.search(results_pattern, text_content, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Also try "Results" heading
    results_heading = soup.find(string=re.compile(r'\d+\s+results?', re.I))
    if results_heading:
        match = re.search(r'(\d+)', results_heading)
        if match:
            return int(match.group(1))
    
    return 0


def extract_all_pgp_ids(base_url):
    """Extract all PGP IDs from all pages of search results."""
    all_pgp_ids = set()
    
    # Get first page to determine total results
    print("Fetching first page to determine total results...")
    pgp_ids, soup = extract_pgp_ids_from_page(base_url, 1)
    all_pgp_ids.update(pgp_ids)
    print(f"Found {len(pgp_ids)} PGP IDs on page 1")
    if pgp_ids:
        print(f"  Sample IDs from page 1: {pgp_ids[:5]}")
    
    # Try to determine total results
    total_results = get_total_results(soup)
    print(f"Total results reported: {total_results}")
    
    # There are 8 pages total
    total_pages = 8
    print(f"Fetching all {total_pages} pages...")
    
    # Fetch all remaining pages (2 through 8)
    for page_num in range(2, total_pages + 1):
        pgp_ids, _ = extract_pgp_ids_from_page(base_url, page_num)
        
        if not pgp_ids:
            print(f"Warning: No PGP IDs found on page {page_num}")
        else:
            all_pgp_ids.update(pgp_ids)
            print(f"Found {len(pgp_ids)} PGP IDs on page {page_num} (total so far: {len(all_pgp_ids)})")
            if len(pgp_ids) <= 5:
                print(f"  IDs from page {page_num}: {pgp_ids}")
        
        time.sleep(1)  # Be polite to the server
    
    return sorted(list(all_pgp_ids))


def main():
    # The search URL
    search_url = "https://geniza.princeton.edu/en/documents/?mode=general&q=Nahray&docdate_0=&docdate_1=&has_transcription=on&doctype=Letter&sort=relevance"
    
    print(f"Extracting PGP IDs from: {search_url}")
    print("=" * 60)
    
    pgp_ids = extract_all_pgp_ids(search_url)
    
    print("=" * 60)
    print(f"\nTotal unique PGP IDs found: {len(pgp_ids)}")
    
    # Save to file
    output_file = "nahray_pgp_ids.txt"
    with open(output_file, 'w') as f:
        for pgp_id in pgp_ids:
            f.write(f"{pgp_id}\n")
    
    print(f"\nSaved PGP IDs to {output_file}")
    
    # Also save as JSON for easy programmatic access
    json_file = "nahray_pgp_ids.json"
    with open(json_file, 'w') as f:
        json.dump(pgp_ids, f, indent=2)
    
    print(f"Also saved as JSON to {json_file}")
    
    # Print first few as sample
    if pgp_ids:
        print(f"\nFirst 10 PGP IDs: {pgp_ids[:10]}")


if __name__ == "__main__":
    main()


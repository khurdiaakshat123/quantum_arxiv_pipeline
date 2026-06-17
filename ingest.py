#!/usr/bin/env python3
"""
arXiv Data Ingestion Pipeline for Quantum Networks

This script queries the official arXiv API for the latest 10 papers related to:
- Quantum Networks
- Network Topology
- Network Congestion

It parses the Atom (XML) response, extracts metadata (title, authors, publish date,
abstract, PDF URL), prints a clean report to the terminal, and downloads the PDFs
into a local directory with rate-limiting and error-handling safeguards.
"""

import os
import re
import time
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import shutil

# Resolve absolute paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPERS_DIR = os.path.join(SCRIPT_DIR, "quantum_papers")
LOG_FILE = os.path.join(SCRIPT_DIR, "ingestion.log")

# Configure logger to write to both stdout and a log file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8')
    ]
)
logger = logging.getLogger("arxiv_ingest")

# XML namespaces used by the arXiv Atom feed
NAMESPACES = {
    'atom': 'http://www.w3.org/2005/Atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/',
    'arxiv': 'http://arxiv.org/schemas/atom'
}


def sanitize_filename(title: str) -> str:
    """
    Sanitizes a string to be used as a safe filename.
    Removes invalid characters and replaces whitespace with underscores.
    """
    # Keep alphanumeric characters, spaces, hyphens, and underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', "", title)
    # Replace any sequence of whitespace with a single underscore
    sanitized = re.sub(r'\s+', "_", sanitized)
    # Strip leading/trailing underscores or spaces
    sanitized = sanitized.strip("_ ")
    # Truncate to avoid path length limits (max 120 chars)
    if len(sanitized) > 120:
        sanitized = sanitized[:117] + "..."
    return sanitized


def build_arxiv_url() -> str:
    """
    Constructs the arXiv API query URL for the latest 10 papers matching the criteria.
    """
    # Query terms matching Quantum Networks, Network Topology, or Network Congestion
    query_parts = [
        'all:"Quantum Networks"',
        'all:"Network Topology"',
        'all:"Network Congestion"'
    ]
    # Join terms with OR boolean operator
    search_query = " OR ".join(query_parts)
    
    params = {
        'search_query': search_query,
        'start': 0,
        'max_results': 10,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }
    
    encoded_params = urllib.parse.urlencode(params)
    url = f"https://export.arxiv.org/api/query?{encoded_params}"
    logger.debug(f"Built arXiv API URL: {url}")
    return url


def fetch_metadata(url: str) -> str:
    """
    Fetches the XML response from the arXiv API.
    """
    logger.info("Querying arXiv API for latest papers...")
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'QuantumNetworkIngestionPipeline/1.0 (student project)'}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to fetch metadata from arXiv: {e}")
        raise


def parse_arxiv_response(xml_data: str) -> list:
    """
    Parses the Atom XML feed and extracts list of papers with metadata.
    """
    logger.info("Parsing XML response...")
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error(f"XML parse error: {e}")
        raise ValueError("Invalid XML response received from arXiv.")

    papers = []
    
    # Find all <entry> elements representing individual papers
    entries = root.findall('atom:entry', NAMESPACES)
    logger.info(f"Found {len(entries)} papers in the feed.")

    for entry in entries:
        # Extract title
        title_elem = entry.find('atom:title', NAMESPACES)
        title = title_elem.text.strip() if title_elem is not None else "Untitled"
        # Clean internal newlines or multiple spaces within titles
        title = re.sub(r'\s+', ' ', title)

        # Extract published date
        published_elem = entry.find('atom:published', NAMESPACES)
        published_date = published_elem.text.strip() if published_elem is not None else "Unknown Date"

        # Extract summary / abstract
        summary_elem = entry.find('atom:summary', NAMESPACES)
        summary = summary_elem.text.strip() if summary_elem is not None else ""
        summary = re.sub(r'\s+', ' ', summary)

        # Extract authors
        authors = []
        for author in entry.findall('atom:author', NAMESPACES):
            name_elem = author.find('atom:name', NAMESPACES)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        
        # Extract PDF URL
        pdf_url = None
        # Look for a link specifically targeting the pdf title or type
        for link in entry.findall('atom:link', NAMESPACES):
            rel = link.attrib.get('rel', '')
            link_type = link.attrib.get('type', '')
            title_attr = link.attrib.get('title', '')
            href = link.attrib.get('href', '')
            
            if title_attr == 'pdf' or link_type == 'application/pdf':
                pdf_url = href
                break
                
        # Fallback 1: check link containing /pdf/
        if not pdf_url:
            for link in entry.findall('atom:link', NAMESPACES):
                href = link.attrib.get('href', '')
                if '/pdf/' in href:
                    pdf_url = href
                    break
                    
        # Fallback 2: convert /abs/ link to /pdf/ link
        if not pdf_url:
            id_elem = entry.find('atom:id', NAMESPACES)
            if id_elem is not None and id_elem.text:
                id_url = id_elem.text.strip()
                if '/abs/' in id_url:
                    pdf_url = id_url.replace('/abs/', '/pdf/')
        
        papers.append({
            'title': title,
            'authors': authors,
            'published': published_date,
            'summary': summary,
            'pdf_url': pdf_url
        })
        
    return papers


def download_paper_pdf(pdf_url: str, title: str, delay_seconds: float = 3.0) -> bool:
    """
    Downloads the paper PDF with basic rate-limiting and robust error handling.
    """
    if not pdf_url:
        logger.warning(f"No PDF URL found for paper: '{title}'. Skipping download.")
        return False

    # Standardize HTTP -> HTTPS for arXiv downloads
    if pdf_url.startswith("http://"):
        pdf_url = "https://" + pdf_url[7:]

    # Sanitize title to create a safe file name
    safe_title = sanitize_filename(title)
    filename = f"{safe_title}.pdf"
    filepath = os.path.join(PAPERS_DIR, filename)

    # Rate limiting: Sleep before initiating download to avoid blocking
    logger.info(f"Rate limiting: waiting {delay_seconds} seconds before downloading...")
    time.sleep(delay_seconds)

    logger.info(f"Downloading PDF: '{title}'")
    logger.info(f"Source URL: {pdf_url}")
    logger.info(f"Target path: {filepath}")

    req = urllib.request.Request(
        pdf_url,
        headers={'User-Agent': 'QuantumNetworkIngestionPipeline/1.0 (student project)'}
    )

    try:
        # Create papers directory if it doesn't exist
        os.makedirs(PAPERS_DIR, exist_ok=True)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get('Content-Type', '')
            # Verify we are actually downloading a PDF file
            if 'pdf' not in content_type.lower() and response.status == 200:
                logger.warning(f"Unexpected Content-Type: '{content_type}'. Attempting to write anyway.")

            with open(filepath, 'wb') as f:
                # Read response in chunks of 8KB to be memory-efficient
                shutil.copyfileobj(response, f)
                
        logger.info(f"Successfully downloaded: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF for '{title}': {e}")
        # Return False to let the caller handle it and continue to the next paper
        return False


def main():
    """
    Runs the full ingestion pipeline.
    """
    logger.info("=" * 60)
    logger.info("Starting arXiv Quantum Networks Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Downloads target directory: {PAPERS_DIR}")
    logger.info(f"Logs file: {LOG_FILE}")
    
    url = build_arxiv_url()
    
    try:
        xml_data = fetch_metadata(url)
        papers = parse_arxiv_response(xml_data)
    except Exception as e:
        logger.critical(f"Ingestion pipeline stopped during metadata fetching/parsing: {e}")
        return

    if not papers:
        logger.warning("No papers found matching the search criteria.")
        return

    downloaded_count = 0
    failed_count = 0

    for idx, paper in enumerate(papers, start=1):
        logger.info("-" * 60)
        logger.info(f"Paper {idx}/{len(papers)}:")
        logger.info(f"Title:     {paper['title']}")
        logger.info(f"Authors:   {', '.join(paper['authors'])}")
        logger.info(f"Published: {paper['published']}")
        logger.info(f"PDF URL:   {paper['pdf_url']}")
        logger.info(f"Abstract:  {paper['summary'][:150]}...")
        
        # Download the paper with rate limiting
        success = download_paper_pdf(paper['pdf_url'], paper['title'], delay_seconds=3.0)
        if success:
            downloaded_count += 1
        else:
            failed_count += 1

    logger.info("=" * 60)
    logger.info("Ingestion Pipeline Summary:")
    logger.info(f"Total papers processed: {len(papers)}")
    logger.info(f"Successfully downloaded: {downloaded_count}")
    logger.info(f"Failed downloads:        {failed_count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

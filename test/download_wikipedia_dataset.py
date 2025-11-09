#!/usr/bin/env python3
# download_wikipedia_dataset.py
# Quick script to upgrade Wikipedia dataset

import json
import wikipedia
from pathlib import Path

def download_wikipedia_dataset(output_file="data/wikipedia_enhanced.json", num_docs=5000):
    """Download diverse Wikipedia articles for better coverage"""
    
    # Comprehensive search queries covering many topics
    search_queries = [
        # Music & Entertainment
        "Eminem rapper", "50 Cent rapper", "Jay-Z", "Rihanna singer",
        "Michael Jackson", "Hip hop music", "Rap music", "Music genre",
        "Record label", "Grammy Award",
        
        # History & Politics
        "American history", "World War II", "Napoleon", "Abraham Lincoln",
        "French Revolution", "Roman Empire", "Medieval period",
        "Modern history", "Cold War",
        
        # Science & Nature
        "Physics", "Chemistry", "Biology", "Quantum mechanics",
        "Albert Einstein", "Marie Curie", "Isaac Newton",
        "Theory of relativity", "Evolution", "Genetics",
        
        # Technology & Engineering
        "Computer", "Internet", "Artificial intelligence",
        "Machine learning", "Python programming", "Algorithm",
        "Data structure", "Software engineering",
        
        # Sports
        "Basketball", "Football", "Soccer", "Baseball",
        "Olympics", "Tennis", "Athletic sports",
        
        # Geography & Culture
        "United States", "Europe", "Asia", "Africa",
        "China", "India", "Brazil", "European culture",
        
        # Literature & Art
        "Shakespeare", "Novel", "Poetry", "Ancient literature",
        "Renaissance", "Art history", "Painting",
        
        # General Knowledge
        "Solar system", "Earth", "Ocean", "Mountain",
        "Desert", "Forest", "Island", "River",
        "Weather", "Climate", "Ecosystem",
        
        # People & Professions
        "President", "Prime Minister", "General", "Inventor",
        "Scientist", "Philosopher", "Author", "Artist",
        
        # Objects & Concepts
        "Bridge", "Building", "Ship", "Train", "Airplane",
        "Car", "Bicycle", "House", "City",
    ]
    
    documents = []
    seen_titles = set()
    
    print("=" * 80)
    print("DOWNLOADING WIKIPEDIA DATASET")
    print("=" * 80)
    print(f"Target: {num_docs} documents")
    print(f"Sources: {len(search_queries)} categories\n")
    
    for query_idx, query in enumerate(search_queries, 1):
        if len(documents) >= num_docs:
            break
        
        print(f"[{query_idx}/{len(search_queries)}] Searching: {query}...")
        
        try:
            # Search Wikipedia
            results = wikipedia.search(query, results=5)
            
            for title in results:
                if len(documents) >= num_docs:
                    break
                
                if title in seen_titles:
                    continue
                
                try:
                    # Get article
                    page = wikipedia.page(title, auto_suggest=False)
                    text = page.content
                    
                    # Only keep substantial articles
                    if len(text) > 200:
                        documents.append(text)
                        seen_titles.add(title)
                        print(f"  ✓ {title[:60]}")
                
                except wikipedia.exceptions.DisambiguationError:
                    continue
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as e:
                    continue
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"Total unique documents: {len(documents)}")
    print(f"Coverage: {(len(documents)/num_docs)*100:.1f}%\n")
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to file
    print(f"Saving to: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(documents),
            'documents': documents,
            'description': 'Enhanced Wikipedia dataset with diverse coverage',
            'sources': len(search_queries)
        }, f, indent=2)
    
    file_size = output_path.stat().st_size / (1024 * 1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    print(f"\n✓ Dataset upgrade complete!")
    
    return len(documents)


if __name__ == "__main__":
    import sys
    
    # Check if wikipedia is installed
    try:
        import wikipedia
    except ImportError:
        print("Error: wikipedia module not found")
        print("Install it with: pip install wikipedia")
        sys.exit(1)
    
    # Get number of docs from command line or use default
    num_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    
    # Download
    try:
        download_wikipedia_dataset(num_docs=num_docs)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

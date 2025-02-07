import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



CLASSES = [
    ["item-title shokz_sans_display","item-subtitle shokz_sans_display", "item-content"],
    "tt-post-content",
    "tt-post-content",
    "tt-post-content"
]

WEB_PATHS = [
    "https://ca.shokz.com/pages/our-story",
    "https://ca.shokz.com/blogs/guides/tips-for-storing-and-maintaining-wireless-earbuds",
    "https://ca.shokz.com/blogs/guides/are-bone-conduction-headphones-safe",
    "https://ca.shokz.com/blogs/guides/earbuds-vs-headphones-which-one-is-better"
]

def load_web_content(web_paths, classes):
    """
    Load content from specified web URLs, keeping only content with specified class names.
    
    Args:
        web_paths (str or list/tuple): Web URLs to load content from
        classes (str or list/tuple): HTML element class names to extract
    
    Returns:
        list: List of loaded documents
        
    Raises:
        ValueError: If web_paths or classes are empty
        requests.RequestException: If web page loading fails
    """
    # Validate inputs
    if not web_paths or not classes:
        raise ValueError("web_paths and classes cannot be empty")
        
    # Ensure web_paths and classes are in tuple format
    if isinstance(web_paths, str):
        web_paths = (web_paths,)
    if isinstance(classes, str):
        classes = (classes,)
        
    try:
        # Create bs4 strainer
        bs4_strainer = bs4.SoupStrainer(class_=classes)
        
        # Create loader and load content
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        return loader.load()
    except Exception as e:
        raise Exception(f"Failed to load web content: {str(e)}")

def main():
    """Main function: Load and process web content"""
    # Load all content using a loop
    all_docs = []
    for web_path, class_names in zip(WEB_PATHS, CLASSES):
        try:
            docs = load_web_content(web_path, class_names)
            all_docs.extend(docs)
            print(f"Successfully loaded content from {web_path}")
        except Exception as e:
            print(f"Failed to load content from {web_path}: {str(e)}")

    print(f"Total documents loaded: {len(all_docs)}")

    # Split all loaded documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(all_docs)

    print(f"Split documents into {len(all_splits)} chunks")

    # Optional: Preview split documents
    for i, split in enumerate(all_splits[:3]):
        print(f"\nChunk {i+1} preview:")
        print(f"Source: {split.metadata['source']}")
        print(f"Content length: {len(split.page_content)} characters")
        print(split.page_content[:100] + "...")

    return all_splits

if __name__ == "__main__":
    main()
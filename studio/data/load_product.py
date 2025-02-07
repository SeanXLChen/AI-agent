import json
from typing import List, Dict, Any, Optional
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from bs4 import BeautifulSoup
import html

def clean_html_content(html_content: str) -> str:
    """Extract readable text from HTML content."""
    decoded_html = html.unescape(html_content)
    soup = BeautifulSoup(decoded_html, 'html.parser')
    text_elements = []
    spans = soup.find_all('span')
    for span in spans:
        if span.string:
            text_elements.append(span.string.strip())
    return '\n'.join(filter(None, dict.fromkeys(text_elements)))

def clean_metadata_value(value: Any) -> Optional[str]:
    """Convert metadata value to string or filter out if None."""
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return str(value)

def create_product_metadata(product: Dict) -> Dict[str, str]:
    """Create metadata dictionary for a product with cleaned values."""
    raw_metadata = {
        "title": product.get('title', ''),
        "vendor": product.get('vendor', ''),
        "product_type": product.get('product_type', ''),
        "created_at": product.get('created_at', ''),
        "updated_at": product.get('updated_at', ''),
        "tags": ','.join(product.get('tags', [])),
        "handle": product.get('handle', ''),
        "product_url": f"https://ca.shokz.com//products/{product.get('handle', '')}"
    }
    
    # Add variant information if available
    if product.get('variants') and len(product['variants']) > 0:
        variant = product['variants'][0]
        raw_metadata.update({
            "price": variant.get('price', ''),
            "sku": variant.get('sku', ''),
            "variant_title": variant.get('title', ''),
            "available": str(variant.get('available', False))
        })
    
    # Add primary image if available
    if product.get('images') and len(product['images']) > 0:
        image = product['images'][0]
        raw_metadata.update({
            "image_src": image.get('src', ''),
            "image_width": str(image.get('width', '')),
            "image_height": str(image.get('height', ''))
        })
    
    # Clean all metadata values
    cleaned_metadata = {
        key: clean_metadata_value(value)
        for key, value in raw_metadata.items()
    }
    
    return cleaned_metadata

def format_product_for_embedding(product: Dict) -> str:
    """Format a product dictionary into a searchable text string."""
    formatted_text = f"""Product: {product.get('title', '')}
Vendor: {product.get('vendor', '')}
Type: {product.get('product_type', '')}
"""
    
    # Clean HTML content if present
    if product.get('body_html'):
        cleaned_text = clean_html_content(product['body_html'])
        if cleaned_text:
            formatted_text += f"Features:\n{cleaned_text}\n"
    
    # Add variants information
    if product.get('variants') and len(product['variants']) > 0:
        variant = product['variants'][0]
        formatted_text += f"""
Price: ${variant.get('price', 'N/A')}
SKU: {variant.get('sku', 'N/A')}
Size/Option: {variant.get('title', 'N/A')}
"""
    
    # Add tags information
    if product.get('tags'):
        benefit_tags = [tag.replace('Benefits:', '') for tag in product['tags'] if tag.startswith('Benefits:')]
        ingredient_tags = [tag.replace('Ingredients:', '') for tag in product['tags'] if tag.startswith('Ingredients:')]
        lifestyle_tags = [tag.replace('Lifestyle:', '') for tag in product['tags'] if tag.startswith('Lifestyle:')]
        
        if benefit_tags:
            formatted_text += f"Benefits: {', '.join(benefit_tags)}\n"
        if ingredient_tags:
            formatted_text += f"Ingredients: {', '.join(ingredient_tags)}\n"
        if lifestyle_tags:
            formatted_text += f"Lifestyle: {', '.join(lifestyle_tags)}\n"
    
    return formatted_text

def load_products_to_vectordb(products_data: List[Dict], 
                            embedding_model: FastEmbedEmbeddings = None,
                            persist_directory: str = "./retailer_db") -> Chroma:
    """Load products into a Chroma vector database."""
    if embedding_model is None:
        embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    documents = []
    ids = []
    metadatas = []
    
    for product in products_data:
        # Skip certain products
        if any(tag in product.get('tags', []) for tag in ['bogos-gift']):
            continue
            
        # Format product text
        formatted_text = format_product_for_embedding(product)
        
        # Create metadata with cleaned values
        metadata = create_product_metadata(product)
        
        # Add to lists
        documents.append(formatted_text)
        ids.append(str(product['id']))
        metadatas.append(metadata)
    
    # Create and populate vector store
    vectorstore = Chroma(
        collection_name="products",
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    
    # Add products to the vectorstore
    vectorstore.add_texts(
        texts=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    return vectorstore

# Example usage
if __name__ == "__main__":
    # Load JSON data
    with open("./data_products.json", "r") as f:
        data = json.load(f)
    
    # Initialize embedding model
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # Load products into vector database
    vectorstore = load_products_to_vectordb(
        data["products"],
        embedding_model=embed_model
    )
    
    # Test a search
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("magnesium supplement for sleep")
    
    for doc in docs:
        print("\n=== Relevant Product ===")
        print(f"Product ID: {doc.id}")
        print(f"Title: {doc.metadata['title']}")
        print(f"Price: ${doc.metadata['price']}")
        print(f"Product URL: {doc.metadata['product_url']}")
        if 'image_src' in doc.metadata:
            print(f"Image: {doc.metadata['image_src']}")
        print("\nDescription:")
        print(doc.page_content)
        print("======================")
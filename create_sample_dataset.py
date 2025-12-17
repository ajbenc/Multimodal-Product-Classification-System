"""
Create a small sample dataset for portfolio demo
This copies a subset of products with images for deployment
"""

import pandas as pd
import shutil
import os
from pathlib import Path

# Configuration
SAMPLE_SIZE_PER_CATEGORY = 3  # Products per category
OUTPUT_DIR = 'demo_sample'

def create_sample_dataset():
    print("📦 Creating sample dataset for portfolio demo...")
    
    # Load full dataset
    df = pd.read_csv('data/processed_products_with_images.csv')
    print(f"Loaded {len(df)} products")
    
    # Get unique categories
    categories = df['class_id'].unique()
    print(f"Found {len(categories)} categories")
    
    # Sample products from each category
    sample_df = df.groupby('class_id').head(SAMPLE_SIZE_PER_CATEGORY)
    print(f"Selected {len(sample_df)} sample products")
    
    # Create output directories
    os.makedirs(f'{OUTPUT_DIR}/images', exist_ok=True)
    
    # Copy images that exist
    copied = 0
    missing = 0
    
    for idx, row in sample_df.iterrows():
        image_path = row['image_path']
        if pd.notna(image_path) and os.path.exists(image_path):
            # Copy image
            filename = os.path.basename(image_path)
            dest = f'{OUTPUT_DIR}/images/{filename}'
            shutil.copy2(image_path, dest)
            
            # Update path in dataframe
            sample_df.at[idx, 'image_path'] = f'images/{filename}'
            copied += 1
        else:
            sample_df.at[idx, 'image_path'] = None
            missing += 1
    
    # Save sample dataset
    sample_df.to_csv(f'{OUTPUT_DIR}/sample_products.csv', index=False)
    
    print(f"\n✅ Sample dataset created!")
    print(f"   📁 Location: {OUTPUT_DIR}/")
    print(f"   📊 Products: {len(sample_df)}")
    print(f"   🖼️  Images copied: {copied}")
    print(f"   ❌ Images missing: {missing}")
    
    # Calculate total size
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    
    size_mb = total_size / (1024 * 1024)
    print(f"   💾 Total size: {size_mb:.1f} MB")
    
    if size_mb < 100:
        print(f"\n✨ This is small enough for GitHub! (< 100MB)")
    else:
        print(f"\n⚠️  Still large for GitHub. Consider reducing SAMPLE_SIZE_PER_CATEGORY")
    
    print(f"\n📝 Next steps:")
    print(f"   1. Update app.py to use '{OUTPUT_DIR}/sample_products.csv'")
    print(f"   2. Test the demo with sample data")
    print(f"   3. Commit the sample to GitHub for deployment")

if __name__ == "__main__":
    create_sample_dataset()

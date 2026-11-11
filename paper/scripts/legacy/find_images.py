import os
import glob
from datetime import datetime
from PIL import Image

def get_image_info():
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.pdf', '*.svg')
    files = []
    for ext in extensions:
        files.extend(glob.glob(f'**/{ext}', recursive=True))
    
    for f in files:
        if 'hf_space' in f or '.git' in f:
            continue
        stat = os.stat(f)
        size_kb = stat.st_size / 1024
        mtime = datetime.fromtimestamp(stat.st_mtime).isoformat()
        dims = 'N/A'
        dpi = 'N/A'
        try:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(f) as img:
                    dims = f'{img.width}x{img.height}'
                    dpi = img.info.get('dpi', 'N/A')
        except Exception as e:
            dims = f'Error: {e}'
        
        print(f'{f} | {size_kb:.2f} KB | {dims} | DPI: {dpi} | Modified: {mtime}')

get_image_info()

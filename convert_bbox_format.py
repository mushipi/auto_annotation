import json
import os
import argparse
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_bbox(bbox):
    """
    Convert [x1, y1, x2, y2] to [x, y, w, h]
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]

def process_file(file_path, dry_run=False):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        if 'objects' in data:
            for obj in data['objects']:
                if 'bbox' in obj:
                    bbox = obj['bbox']
                    if len(bbox) == 4:
                        # Heuristic check: if x2 > x1 and y2 > y1, it might be [x1, y1, x2, y2]
                        # If it is already [x, y, w, h], then usually w and h are positive.
                        # But [x1, y1, x2, y2] also has x2 > x1.
                        # However, if we interpret [x, y, w, h] as [x1, y1, x2, y2], then x2 (w) is likely < x1 (x) for normal objects unless they are at 0,0.
                        # Wait, if bbox is [100, 100, 50, 50] (x,y,w,h), then "x2" (50) < "x1" (100).
                        # So if bbox[2] < bbox[0] or bbox[3] < bbox[1], it is definitely [x, y, w, h] (or invalid).
                        # If bbox[2] > bbox[0] and bbox[3] > bbox[1], it is likely [x1, y1, x2, y2].
                        
                        x1, y1, x2, y2 = bbox
                        if x2 > x1 and y2 > y1:
                            # Likely [x1, y1, x2, y2]
                            new_bbox = convert_bbox(bbox)
                            obj['bbox'] = new_bbox
                            modified = True
                            # logger.debug(f"Converted bbox in {file_path}: {bbox} -> {new_bbox}")
        
        if modified:
            if not dry_run:
                # Backup
                backup_path = str(file_path) + ".bak"
                shutil.copy2(file_path, backup_path)
                
                # Save
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Converted and saved: {file_path}")
            else:
                logger.info(f"[Dry Run] Would convert: {file_path}")
        else:
            logger.debug(f"No conversion needed for: {file_path}")
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert bbox format in JSON files from [x1,y1,x2,y2] to [x,y,w,h]")
    parser.add_argument('--dir', '-d', default='./output/annotations', help='Directory to search for JSON files')
    parser.add_argument('--dry-run', action='store_true', help='Do not write changes, just print what would happen')
    
    args = parser.parse_args()
    
    target_dir = Path(args.dir)
    if not target_dir.exists():
        logger.error(f"Directory not found: {target_dir}")
        return

    logger.info(f"Scanning directory: {target_dir}")
    
    count = 0
    for file_path in target_dir.rglob('*.json'):
        process_file(file_path, dry_run=args.dry_run)
        count += 1
        
    logger.info(f"Processed {count} files.")

if __name__ == "__main__":
    main()

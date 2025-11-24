"""
Re-annotation API Server for auto_annotation

This FastAPI server exposes SAM3 re-annotation functionality as a REST API.
It allows external clients (like annotation_checker) to request re-annotation
without needing to import PyTorch/CUDA dependencies.
"""

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from pathlib import Path
import cv2
import yaml
import logging

# Import from local modules
from src.annotator import SAM3Annotator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Auto-annotation Re-annotation API")

# Global annotator instance (lazy loaded)
annotator_instance = None

class ReannotateRequest(BaseModel):
    image_path: str
    prompt: str | None = None

class ReannotateResponse(BaseModel):
    success: bool
    message: str
    results: dict | None = None

def get_annotator():
    """Lazy load and return the SAM3Annotator instance"""
    global annotator_instance
    
    if annotator_instance is None:
        logger.info("Loading SAM3Annotator...")
        
        # Load config
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Initialize annotator
        annotator_instance = SAM3Annotator(config['annotation'])
        
        # Load model
        success = annotator_instance.load_model()
        if not success:
            logger.error("Failed to load SAM3 model")
            annotator_instance = None
            raise RuntimeError("Failed to load SAM3 model")
        
        logger.info("SAM3Annotator loaded successfully")
    
    return annotator_instance

@app.post("/api/reannotate", response_model=ReannotateResponse)
async def reannotate(request: ReannotateRequest = Body(...)):
    """
    Re-annotate an image with optional custom prompt
    
    Args:
        image_path: Full path to the image file
        prompt: Optional custom prompt (e.g., "person", "car")
    
    Returns:
        ReannotateResponse with success status and results
    """
    try:
        # Validate image path
        image_path = Path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")
        
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to load image")
        
        # Get annotator
        try:
            annotator = get_annotator()
        except Exception as e:
            logger.error(f"Failed to get annotator: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize annotator: {e}")
        
        # Run annotation
        logger.info(f"Annotating {image_path.name} with prompt: {request.prompt or 'default'}")
        results = annotator.annotate(img, prompt=request.prompt)
        
        if results is None:
            raise HTTPException(status_code=500, detail="Annotation failed (internal error)")
        
        # Check if objects were found
        object_count = len(results['boxes'])
        if object_count == 0:
            return ReannotateResponse(
                success=False,
                message="No objects detected with the given prompt.",
                results=None
            )
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            "masks": [mask.tolist() for mask in results["masks"]],
            "boxes": results["boxes"],  # Already converted to list in annotator
            "scores": results["scores"],  # Already converted to float
            "labels": results["labels"]
        }
        
        return ReannotateResponse(
            success=True,
            message=f"Successfully annotated. Found {object_count} objects.",
            results=serializable_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during re-annotation: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "model_loaded": annotator_instance is not None}

if __name__ == "__main__":
    import uvicorn
    
    # Read port from config or use default
    config_path = Path(__file__).parent / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        port = config.get('api', {}).get('port', 8000)
        host = config.get('api', {}).get('host', '127.0.0.1')
    else:
        port = 8000
        host = '127.0.0.1'
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

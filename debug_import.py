import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

print("\nAttempting to import sam3...")
try:
    import sam3
    print(f"Successfully imported sam3 from: {sam3.__file__}")
    
    from sam3.model_builder import build_sam3_image_model
    print("Successfully imported build_sam3_image_model")
    
except ImportError as e:
    print(f"\nImportError: {e}")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    import traceback
    traceback.print_exc()

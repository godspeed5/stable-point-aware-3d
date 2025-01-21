import os
import io
import uuid
import time
import tempfile
import zipfile
from typing import Dict
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
import trimesh
import torch

# Import functions and utilities from your reference code
from gradio_app import run_model

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Unique ID to ZIP path map
zip_map: Dict[str, str] = {}

# FastAPI app
app = FastAPI(
    title="SPAR3D API",
    description="Endpoints to generate and edit 3D models with unique download links.",
    version="1.0.0",
)

# Utility function to save results to ZIP
def save_results_to_zip(
    glb_file: str, pc_file: str, illumination_file: str
) -> str:
    unique_id = str(uuid.uuid4())
    zip_dir = os.path.join(OUTPUT_DIR, unique_id)
    os.makedirs(zip_dir, exist_ok=True)

    zip_path = os.path.join(zip_dir, "output.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(glb_file, arcname="mesh.glb")
        zf.write(pc_file, arcname="points.ply")
        zf.write(illumination_file, arcname="illumination.hdr")

    zip_map[unique_id] = zip_path
    return unique_id

# Endpoints
@app.post("/generate")
async def generate(
    file: UploadFile,
    guidance_scale: float = Form(3.0),
    random_seed: int = Form(0),
    remesh_option: str = Form("none"),
    vertex_count: int = Form(-1),
    texture_resolution: int = Form(1024),
):
    """
    Generate a 3D model from an uploaded image.
    """
    start_time = time.time()

    # Read and prepare input image
    img_bytes = await file.read()
    input_image = Image.open(io.BytesIO(img_bytes))

    # Call `run_model` from your reference code
    glb_file, pc_file, illumination_file, _ = run_model(
        input_image=input_image,
        guidance_scale=guidance_scale,
        random_seed=random_seed,
        pc_cond=None,
        remesh_option=remesh_option,
        vertex_count=vertex_count,
        texture_resolution=texture_resolution,
    )

    # Save results to ZIP
    unique_id = save_results_to_zip(glb_file, pc_file, illumination_file)
    print(f"Generation completed in {time.time() - start_time:.2f} seconds.")

    return JSONResponse(
        {"unique_id": unique_id, "download_url": f"/download/{unique_id}"}
    )


@app.post("/edit")
async def edit(
    img_file: UploadFile,
    pc_file: UploadFile,
    guidance_scale: float = Form(3.0),
    random_seed: int = Form(0),
    remesh_option: str = Form("none"),
    vertex_count: int = Form(-1),
    texture_resolution: int = Form(1024),
):
    """
    Re-generate a 3D model using an edited point cloud and the original image.
    """
    start_time = time.time()

    # Read input image
    img_bytes = await img_file.read()
    input_image = Image.open(io.BytesIO(img_bytes))

    # Read edited point cloud
    pc_bytes = await pc_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ply") as tmp:
        tmp.write(pc_bytes)
        tmp.flush()
        edited_pc = trimesh.load(tmp.name)

    # Call `run_model` from your reference code with the edited point cloud
    glb_file, pc_file, illumination_file, _ = run_model(
        input_image=input_image,
        guidance_scale=guidance_scale,
        random_seed=random_seed,
        pc_cond=edited_pc,
        remesh_option=remesh_option,
        vertex_count=vertex_count,
        texture_resolution=texture_resolution,
    )

    # Save results to ZIP
    unique_id = save_results_to_zip(glb_file, pc_file, illumination_file)
    print(f"Editing completed in {time.time() - start_time:.2f} seconds.")

    return JSONResponse(
        {"unique_id": unique_id, "download_url": f"/download/{unique_id}"}
    )


@app.get("/download/{unique_id}")
def download(unique_id: str):
    """
    Download the generated ZIP file using its unique ID.
    """
    if unique_id not in zip_map:
        raise HTTPException(status_code=404, detail="File not found.")
    zip_path = zip_map[unique_id]
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file missing.")
    return FileResponse(zip_path, media_type="application/zip", filename="output.zip")


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

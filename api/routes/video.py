# api/routes/video.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import uuid
import asyncio

from config.settings import settings
from api.models.response import VideoProcessResponse, BatchProcessResponse
from api.services.recognition import RecognitionService
from api.services.storage import StorageManager

router = APIRouter()

@router.post("/process", response_model=VideoProcessResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Process single video"""
    
    if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid video file")
    
    contents = await file.read()
    if len(contents) > settings.api.max_upload_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    upload_id = str(uuid.uuid4())
    upload_path = StorageManager.save_upload(upload_id, file.filename, contents)
    
    try:
        output_path = RecognitionService.process_video(upload_path, upload_id)
        video_info = StorageManager.get_video_info(upload_path)
        
        background_tasks.add_task(StorageManager.cleanup_upload, upload_path)
        
        return VideoProcessResponse(
            success=True,
            upload_id=upload_id,
            output_url=f"/outputs/{output_path.name}",
            info=video_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-process")
async def batch_process(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Process multiple videos"""
    
    results = []
    
    for file in files:
        try:
            # Validate
            if not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": "Invalid format"
                })
                continue
            
            # Process
            contents = await file.read()
            upload_id = str(uuid.uuid4())
            upload_path = StorageManager.save_upload(upload_id, file.filename, contents)
            
            output_path = RecognitionService.process_video(upload_path, upload_id)
            video_info = StorageManager.get_video_info(upload_path)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "upload_id": upload_id,
                "output_url": f"/outputs/{output_path.name}",
                "info": video_info
            })
            
            # Cleanup
            if background_tasks:
                background_tasks.add_task(StorageManager.cleanup_upload, upload_path)
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total": len(files),
        "processed": len([r for r in results if r.get("success")]),
        "results": results
    }

@router.get("/download/{file_id}")
async def download_video(file_id: str):
    """Download processed video"""
    file_path = settings.data_dir / "outputs" / f"output_{file_id}.mp4"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="video/mp4",
        filename=file_path.name
    )

@router.get("/list-outputs")
async def list_outputs():
    """List all processed videos"""
    output_dir = settings.data_dir / "outputs"
    files = []
    
    for file_path in output_dir.glob("*.mp4"):
        files.append({
            "filename": file_path.name,
            "size": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime
        })
    
    return {
        "success": True,
        "count": len(files),
        "files": files
    }

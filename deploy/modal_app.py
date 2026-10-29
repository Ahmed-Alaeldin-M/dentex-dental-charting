"""Modal deployment: CPU-only FastAPI endpoint for DENTEX dental charting (ONNX Runtime).

Setup (once):
    modal volume create dentex-models
    modal volume put dentex-models teeth_yolov8s.onnx /teeth_yolov8s.onnx
    modal volume put dentex-models diag_yolov8s.onnx /diag_yolov8s.onnx
Deploy:
    modal deploy deploy/modal_app.py
"""
from pathlib import Path

import modal

APP_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "onnxruntime>=1.22",  # models were exported with a recent onnx (IR v11), older ORT can't load them
        "opencv-python-headless>=4.10",
        "numpy",
        "fastapi[standard]",
        "python-multipart",
    )
    .add_local_file(APP_DIR / "inference.py", "/root/inference.py")
)

app = modal.App("dentex-dental-charting", image=image)
models = modal.Volume.from_name("dentex-models", create_if_missing=True)


@app.cls(cpu=2.0, memory=2048, volumes={"/models": models}, scaledown_window=300, max_containers=3)
class DentalCharting:
    @modal.enter()
    def load(self):
        from inference import DIAG_FALLBACK, FDI, OnnxDetector

        self.teeth = OnnxDetector("/models/teeth_yolov8s.onnx", FDI)
        self.diag = OnnxDetector("/models/diag_yolov8s.onnx", DIAG_FALLBACK)

    @modal.asgi_app()
    def web(self):
        import time

        import cv2
        from fastapi import FastAPI, File, HTTPException, Query, UploadFile
        from fastapi.responses import Response

        from inference import chart, draw

        api = FastAPI(title="DENTEX Dental Charting", version="1.0",
                      description="Tooth numbering (FDI) and diagnosis on panoramic dental X-rays. Research demo, not a medical device.")

        def run(data, conf, iou):
            try:
                return chart(self.teeth, self.diag, data, conf=conf, iou=iou)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api.get("/health")
        def health():
            return {"status": "ok", "teeth_classes": len(self.teeth.names), "diagnosis_classes": self.diag.names}

        @api.post("/predict")
        async def predict(file: UploadFile = File(...),
                          conf: float = Query(0.25, ge=0.01, le=0.99),
                          iou: float = Query(0.7, ge=0.1, le=0.95)):
            t0 = time.perf_counter()
            img, teeth, findings = run(await file.read(), conf, iou)
            return {
                "image_size": {"width": img.shape[1], "height": img.shape[0]},
                "teeth_detected": len(teeth),
                "chart": [f"Tooth {f['tooth']}: {f['diagnosis']} ({f['confidence']:.2f})" for f in findings],
                "findings": findings,
                "teeth": teeth,
                "inference_ms": round((time.perf_counter() - t0) * 1000),
            }

        @api.post("/predict/image")
        async def predict_image(file: UploadFile = File(...),
                                conf: float = Query(0.25, ge=0.01, le=0.99),
                                iou: float = Query(0.7, ge=0.1, le=0.95)):
            img, teeth, findings = run(await file.read(), conf, iou)
            ok, buf = cv2.imencode(".jpg", draw(img, teeth, findings), [cv2.IMWRITE_JPEG_QUALITY, 85])
            return Response(content=buf.tobytes(), media_type="image/jpeg")

        return api

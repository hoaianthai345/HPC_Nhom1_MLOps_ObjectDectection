import io
import logging
import requests
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def format_detections_text(result_dict: Dict[str, Any]) -> str:
    """Format detection results as readable text."""
    lines = [
        f"📊 **Detection Results**",
        f"",
        f"- **Objects Found:** {result_dict['num_detections']}",
        f"- **Image Size:** {result_dict['image_size']['width']}x{result_dict['image_size']['height']}",
        f"- **Inference Time:** {result_dict['inference_time_ms']:.2f}ms",
        f"",
    ]
    
    if result_dict['detections']:
        lines.append("**Detected Objects:**")
        lines.append("")
        
        # Group by class
        class_counts = {}
        for det in result_dict['detections']:
            cls_name = det['class_name']
            if cls_name not in class_counts:
                class_counts[cls_name] = []
            class_counts[cls_name].append(det)
        
        for cls_name, dets in sorted(class_counts.items()):
            lines.append(f"**{cls_name}** ({len(dets)} detected):")
            for i, det in enumerate(dets, 1):
                bbox = det['bbox']
                lines.append(
                    f"  {i}. Confidence: {det['confidence']:.2%} | "
                    f"Box: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - "
                    f"({bbox['x2']:.0f}, {bbox['y2']:.0f})"
                )
            lines.append("")
    else:
        lines.append("*No objects detected*")
    
    return "\n".join(lines)


def draw_boxes_on_image(image: Image.Image, result_dict: Dict[str, Any]) -> Image.Image:
    """Vẽ bounding boxes lên ảnh gốc từ kết quả JSON /detect."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    detections = result_dict.get("detections", [])
    if not detections:
        return img

    colors = [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    ]
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14) # Load font in linux 
    except (OSError, TypeError):
        font = ImageFont.load_default() # other os use default font

    for det in detections:
        bbox = det["bbox"]
        x1, y1 = int(bbox["x1"]), int(bbox["y1"])
        x2, y2 = int(bbox["x2"]), int(bbox["y2"])
        class_id = det.get("class_id", 0) % len(colors)
        color = colors[class_id]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        # Vẽ box + label
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.rectangle([x1, y1 - 20, x1 + 200, y1], fill=color)
        draw.text((x1 + 2, y1 - 18), label, fill="white", font=font)

    return img


def call_api(
    image: Image.Image,
    confidence_threshold: float,
    iou_threshold: float,
    use_gpu: bool = False
) -> Tuple[Optional[Image.Image], str]:

    if image is None:
        return None, "Please upload an image"

    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)

        # Use GPU service on different port if GPU is selected, otherwise use CPU service
        if use_gpu:
            api_url = f"http://{settings.GPU_API_HOST}:{settings.GPU_API_PORT}"
        else:
            api_url = f"http://{settings.API_HOST}:{settings.API_PORT}"
        
        endpoint = "/detect"
        
        response = requests.post(
            f"{api_url}{endpoint}",
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")},
            params={
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
            },
            timeout=30,
        )

        if response.status_code != 200:
            return None, f" **API Error:** {response.text}"

        result_dict = response.json()
        results_text = format_detections_text(result_dict)
        annotated_image = draw_boxes_on_image(image, result_dict)
        return annotated_image, results_text

    except requests.exceptions.ConnectionError:
        return None, " **Error:** Cannot connect to API. Make sure the FastAPI server is running."
    except Exception as e:
        logger.error(f"API call error: {str(e)}")
        return None, f" **Error:** {str(e)}"


def check_gpu_available() -> bool:
    """Check if GPU service is available (separate service on different port)."""
    try:
        gpu_api_url = f"http://{settings.GPU_API_HOST}:{settings.GPU_API_PORT}"
        response = requests.get(f"{gpu_api_url}/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        logger.warning("GPU service not reachable at %s:%s", settings.GPU_API_HOST, settings.GPU_API_PORT)
    return False


def create_gradio_app() -> gr.Blocks:
    """Tạo Gradio UI, luôn gọi FastAPI backend (/detect)."""
    # Check GPU availability at startup
    gpu_available = check_gpu_available()
    if gpu_available:
        gpu_status_msg = f"🚀 GPU service is available on port {settings.GPU_API_PORT}"
    else:
        gpu_status_msg = f"⚠️ GPU service not available on port {settings.GPU_API_PORT} (CPU only mode)"
    
    with gr.Blocks(
        title=settings.APP_NAME,
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown(
            f"""
            # 🔍 {settings.APP_NAME}
            
            Upload an image to detect objects using YOLO model.
            
            **Status:** {gpu_status_msg}
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="📤 Upload Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=400
                )
                
                with gr.Accordion("⚙️ Settings", open=True):
                    # Only show GPU option if available
                    if gpu_available:
                        device_choices = [("🖥️ CPU (Standard)", False), ("🚀 GPU (Fast)", True)]
                        device_info = "Select inference device"
                    else:
                        device_choices = [("🖥️ CPU (Standard)", False)]
                        device_info = "GPU not available - using CPU only"
                    
                    device_radio = gr.Radio(
                        choices=device_choices,
                        value=False,
                        label="Inference Device",
                        info=device_info
                    )
                    
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=settings.YOLO_CONFIDENCE_THRESHOLD,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum confidence for a detection to be shown"
                    )
                    
                    iou_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=settings.YOLO_IOU_THRESHOLD,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="Higher value = more overlapping boxes allowed"
                    )
                
                detect_btn = gr.Button(
                    "🔍 Detect Objects",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output section
                output_image = gr.Image(
                    label="🎯 Detection Results",
                    type="pil",
                    height=400
                )
                
                results_text = gr.Markdown(
                    label="📊 Results Summary",
                    value="*Upload an image and click 'Detect Objects'*"
                )
        
        # Info section
        with gr.Accordion("ℹ️ About", open=False):
            gr.Markdown(
                f"""
                ### Configuration
                - **Model:** {settings.YOLO_MODEL_PATH}
                - **Max File Size:** {settings.MAX_FILE_SIZE_MB} MB
                - **Allowed Formats:** {', '.join(settings.ALLOWED_EXTENSIONS)}
                
                ### API Endpoints
                - **CPU Service:** [http://localhost:{settings.API_PORT}/detect](http://localhost:{settings.API_PORT}/detect)
                - **GPU Service:** [http://localhost:{settings.GPU_API_PORT}/detect](http://localhost:{settings.GPU_API_PORT}/detect)
                - **CPU Swagger UI:** [http://localhost:{settings.API_PORT}/docs](http://localhost:{settings.API_PORT}/docs)
                - **GPU Swagger UI:** [http://localhost:{settings.GPU_API_PORT}/docs](http://localhost:{settings.GPU_API_PORT}/docs)
                """
            )
        
        detect_btn.click(
            fn=call_api,
            inputs=[input_image, confidence_slider, iou_slider, device_radio],
            outputs=[output_image, results_text]
        )
    
    return app


def launch():
    """Chạy Gradio UI (frontend cho FastAPI)."""
    app = create_gradio_app()
    app.launch(
        server_name=settings.HOST,
        server_port=settings.GRADIO_PORT,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch Gradio Object Detection UI")
    parser.add_argument(
        "--port",
        type=int,
        default=settings.GRADIO_PORT,
        help="Port to run Gradio on"
    )
    args = parser.parse_args()
    if args.port != settings.GRADIO_PORT:
        settings.GRADIO_PORT = args.port
    launch()

import io
import logging
import requests
from typing import Optional, Tuple, Dict, Any
from PIL import Image
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


def call_api(
    image: Image.Image,
    confidence_threshold: float,
    iou_threshold: float
) -> Tuple[Optional[Image.Image], str]:
    """
    Call the FastAPI backend instead of direct inference.
    """
    if image is None:
        return None, "⚠️ Please upload an image"
    
    try:
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="JPEG", quality=95)
        img_byte_arr.seek(0)
        
        # Call API
        api_url = f"http://{settings.HOST}:{settings.API_PORT}"
        
        # Get annotated image
        response = requests.post(
            f"{api_url}/detect/annotated",
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")},
            params={
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return None, f"❌ **API Error:** {response.text}"
        
        # Get the annotated image
        annotated_image = Image.open(io.BytesIO(response.content))
        
        # Get detection results
        img_byte_arr.seek(0)
        json_response = requests.post(
            f"{api_url}/detect",
            files={"file": ("image.jpg", img_byte_arr, "image/jpeg")},
            params={
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold
            },
            timeout=30
        )
        
        if json_response.status_code == 200:
            result_dict = json_response.json()
            results_text = format_detections_text(result_dict)
        else:
            results_text = "Detection successful (see image)"
        
        return annotated_image, results_text
        
    except requests.exceptions.ConnectionError:
        return None, "❌ **Error:** Cannot connect to API. Make sure the FastAPI server is running."
    except Exception as e:
        logger.error(f"API call error: {str(e)}")
        return None, f"❌ **Error:** {str(e)}"


def create_gradio_app(use_api: bool = True) -> gr.Blocks:
    """
    Create and return the Gradio interface.
    
    Args:
        use_api: If True, calls FastAPI backend. If False, runs inference directly.
        
    Returns:
        Gradio Blocks app
    """
    with gr.Blocks(
        title=settings.APP_NAME,
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown(
            f"""
            # 🔍 {settings.APP_NAME}
            
            Upload an image to detect objects using YOLO model.
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
                - **Swagger UI:** [http://localhost:{settings.API_PORT}/docs](http://localhost:{settings.API_PORT}/docs)
                """
            )
        
        # Set up event handler
        # Always use FastAPI backend; Gradio chỉ là frontend
        detect_fn = call_api
        
        detect_btn.click(
            fn=detect_fn,
            inputs=[input_image, confidence_slider, iou_slider],
            outputs=[output_image, results_text]
        )
    
    return app


def launch_standalone():
    """Launch Gradio app as standalone (direct inference)."""
    app = create_gradio_app(use_api=False)
    app.launch(
        server_name=settings.HOST,
        server_port=settings.GRADIO_PORT,
        share=False,
        show_error=True
    )


def launch_with_api():
    """Launch Gradio app as frontend to FastAPI."""
    app = create_gradio_app(use_api=True)
    app.launch(
        server_name=settings.HOST,
        server_port=settings.GRADIO_PORT,
        share=False,
        show_error=True
    )


# Main entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio Object Detection UI")
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use FastAPI backend instead of direct inference"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.GRADIO_PORT,
        help="Port to run Gradio on"
    )
    
    args = parser.parse_args()
    
    if args.port != settings.GRADIO_PORT:
        settings.GRADIO_PORT = args.port
    
    if args.use_api:
        launch_with_api()
    else:
        launch_standalone()

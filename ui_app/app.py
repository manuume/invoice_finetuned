import gradio as gr
import httpx
import json

FASTAPI_URL = "[http://127.0.0.1:8001/extract/](http://127.0.0.1:8001/extract/)"

def extract_receipt_data(image_file):
    if image_file is None:
        return None

    with httpx.Client() as client:
        try:
            files = {"file": (image_file.name, open(image_file.name, "rb"), "image/jpeg")}
            response = client.post(FASTAPI_URL, files=files, timeout=120.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return f"Error from API: {e.response.text}"
        except httpx.RequestError as e:
            return f"Error connecting to API: {e}"

with gr.Blocks(theme=gr.themes.Soft(), title="Receipt Extractor") as demo:
    gr.Markdown("# Receipt Information Extractor")
    gr.Markdown("Upload a receipt image to extract its data using our fine-tuned AI model.")

    with gr.Row(equal_height=True):
        image_input = gr.Image(type="filepath", label="Upload Receipt Image")
        json_output = gr.JSON(label="Extracted Information")

    submit_btn = gr.Button("Extract Data", variant="primary")
    submit_btn.click(fn=extract_receipt_data, inputs=image_input, outputs=json_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
import spaces
import gradio as gr
import os
import random
import json
import uuid
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, AutoPipelineForText2Image, DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import torch
from typing import Tuple
from datetime import datetime
import requests
import torch
from diffusers import DiffusionPipeline
import importlib

MAX_SEED = 12211231
CACHE_EXAMPLES = "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "4192"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "1") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

NUM_IMAGES_PER_PROMPT = 1


cfg = json.load(open("app.conf"))

def load_pipeline_and_scheduler():
    clip_skip = cfg.get("clip_skip", 0)

    # Download the model files
    ckpt_dir = snapshot_download(repo_id=cfg["model_id"])

    # Load the models
    vae = AutoencoderKL.from_pretrained(os.path.join(ckpt_dir, "vae"), torch_dtype=torch.float16)
   
    pipe = StableDiffusionXLPipeline.from_pretrained(
        ckpt_dir,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    # Define samplers
    samplers = {
        "Euler a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
        "DPM++ SDE Karras": DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    }
    # Set the scheduler based on the selected sampler
    pipe.scheduler = samplers[cfg.get("sampler","DPM++ SDE Karras")]
    
    # Set clip skip
    pipe.text_encoder.config.num_hidden_layers -= (clip_skip - 1)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        print("Model Compiled!")
    return pipe
pipe = load_pipeline_and_scheduler()
css = '''
.gradio-container{max-width: 560px !important}
body {
    background-color: rgb(3, 7, 18);
    color: white;
}
.gradio-container {
    background-color: rgb(3, 7, 18) !important;
    border: none !important;
}
''' 
js = '''
<script src="/prompts.js"></script>
<script>
window.g=function(){ 
  const conditions = {
    "tag": ["normal", "sexy", "porn"],
    "exclude_category": ["Clothing"],
    "count_per_tag": 1
  };
  prompt = generateSexyPrompt()
  console.log(prompt);
  return prompt
} 
window.postMessageToParent = function(prompt, event, source, value) {
    // Construct the message object with the provided parameters
    console.log("post start",event, source, value);
    const message = {
        event: event,
        source: source,
        value: value
    };
    
    // Post the message to the parent window
    window.parent.postMessage(message, '*');
    console.log("post finish");
    return prompt;
}
function uploadImage(prompt, images, event, source, value) {
    // Ensure we're in an iframe
    console.log("uploadImage", prompt, images && images.length > 0 ? images[0].image.url : null, event, source, value);
    if (window.self !== window.top) {
        // Get the first image from the gallery (assuming it's an array)
        let imageUrl = images && images.length > 0 ? images[0].image.url : null;
        
        // Prepare the data to send
        let data = {
            event: event,
            source: source,
            prompt: prompt,
            image: imageUrl
        };
        
        // Post the message to the parent window
        window.parent.postMessage(JSON.stringify(data), '*');
    } else {
        console.log("Not in an iframe, can't post to parent");
    }
}
</script>
'''
def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name
    
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

@spaces.GPU(duration=60)
def generate(prompt, progress=gr.Progress(track_tqdm=True)):
    negative_prompt = cfg.get("negative_prompt", "")
    style_selection = ""
    use_negative_prompt = True
    seed = 0
    width = cfg.get("width", 1024)
    height = cfg.get("width", 768) 
    inference_steps = cfg.get("inference_steps", 30)
    randomize_seed = True
    guidance_scale = cfg.get("guidance_scale", 7.5)
    prompt_str = cfg.get("prompt", "{prompt}").replace("{prompt}", prompt)
      
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator(pipe.device).manual_seed(seed)
        
    images = pipe(
        prompt=prompt_str,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=inference_steps,
        generator=generator,
        num_images_per_prompt=NUM_IMAGES_PER_PROMPT,
        output_type="pil",
    ).images
    
    image_paths = [save_image(img) for img in images]
    print(image_paths)
    return image_paths

    
with gr.Blocks(css=css,head=js,fill_height=True) as demo:
    with gr.Row(equal_height=False):
        with gr.Group():                
            result = gr.Gallery(value=cfg.get("cover_path",""),
              label="Result",  show_label=False, columns=1, rows=1, show_share_button=True,
              show_download_button=True,allow_preview=True,interactive=False, min_width=cfg.get("window_min_width", 340)
            )
            with gr.Row(): 
                prompt = gr.Text(
                    show_label=False,
                    max_lines=2,
                    lines=2,
                    placeholder="Enter what you want to see",
                    container=False,
                    scale=5,
                    min_width=100,
                )
                random_button = gr.Button("Surprise Me", scale=1, min_width=10)
                run_button = gr.Button( "GO!", scale=1, min_width=20)

    random_button.click(fn=lambda x:x, inputs=[prompt], outputs=[prompt], js='''()=>window.g()''')
    run_button.click(generate, inputs=[prompt], outputs=[result], js=f'''(p)=>window.postMessageToParent(p,"process_started","demo_hf_{cfg.get("name")}_card", "click_go")''')
    result.change(fn=lambda x:x, inputs=[prompt,result], outputs=[], js=f'''(p,img)=>window.uploadImage(p, img,"process_started","demo_hf_{cfg.get("name")}_card", "finish")''')
    
if __name__ == "__main__":
    demo.queue(max_size=200).launch()

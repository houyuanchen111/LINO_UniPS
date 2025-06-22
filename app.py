# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] [Chongjie Ye] 
# SPDX-License-Identifier: MIT
# This file has been modified by Chongjie Ye on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.

import gradio as gr
import os
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
from Stable3DGen.hi3dgen.pipelines import Hi3DGenPipeline
import trimesh
import tempfile
from PIL import Image
import glob
from src.data import DemoData
from src.models import LiNo_UniPS
from torch.utils.data import DataLoader
import pytorch_lightning as pl


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def cache_weights(weights_dir: str) -> dict:
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "houyuanchen/lino"
    ]
    cached_paths = {}
    for model_id in model_ids:
        print(f"Caching weights for: {model_id}")
        # Check if the model is already cached
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            print(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            continue
        # Download the model and cache it
        print(f"Downloading and caching model: {model_id}")
        # Use snapshot_download to download the model
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        print(f"Cached at: {local_path}")

    return cached_paths

def preprocess_mesh(mesh_prompt):
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'

def generate_3d(image, seed=-1,  
                ss_guidance_strength=3, ss_sampling_steps=50,
                slat_guidance_strength=3, slat_sampling_steps=6,normal_bridge=None):
    if image is None:
        return None, None, None

    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
    
    # image = hi3dgen_pipeline.preprocess_image(image, resolution=1024)
    # normal_image = normal_predictor(image, resolution=768, match_input_resolution=True, data_type='object')
    if normal_bridge is None:
        return 0 
    mask = np.float32(np.abs(1 - np.sqrt(np.sum(normal_bridge * normal_bridge, axis=2))) < 0.5)[:,:,None]
    normal_image = mask * (normal_bridge * 0.5 + 0.5)
    normal_image = np.concatenate((normal_image,mask),axis=2)*255.0
    normal_image = Image.fromarray(normal_image.astype(np.uint8),mode="RGBA") 


    outputs = hi3dgen_pipeline.run(
        normal_image,
        seed=seed,
        formats=["mesh",],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    generated_mesh = outputs['mesh'][0]
    
    # Save outputs
    import datetime
    output_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(os.path.join(TMP_DIR, output_id), exist_ok=True)
    mesh_path = f"{TMP_DIR}/{output_id}/mesh.glb"
    
    # Export mesh
    trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)

    trimesh_mesh.export(mesh_path)

    return mesh_path, mesh_path

def predict_normal(input_images,input_mask):
    test_dataset = DemoData(input_imgs_list=input_images,input_mask=input_mask)
    test_loader = DataLoader(test_dataset, batch_size=1)

    trainer = pl.Trainer(accelerator="auto", devices=1,precision="bf16")
    nml_predict = trainer.predict(model=lino, dataloaders=test_loader)

    nml_output = 0.5 * nml_predict[0] + 0.5
    
    return ((nml_output*255.0).astype(np.uint8), nml_predict[0])

def load_example_data(path):
    path = os.path.join("demo", path)
    mask_path = os.path.join(path,"mask.png")
    image_pathes = glob.glob(os.path.join(path, f"L*")) + glob.glob(os.path.join(path, f"0*"))
    input_images = []
    for p in image_pathes:
        input_images.append(Image.open(p))
    
    if os.path.exists(mask_path):
        input_mask = Image.open(mask_path)
    else:
        input_mask =Image.fromarray(np.ones_like(np.array(input_images[0])))
    return input_mask, input_images

# Create the Gradio interface with improved layout
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Light of Normals: Unified Feature Representation for Universal Photometric Stereo</h1>
        """
    )
    
    with gr.Row():
        gr.Markdown("""
                    <p align="center">
                    <a title="Website" href="https://houyuanchen111.github.io/lino.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/Github-Page-black">
                    </a>
                    <a title="Github" href="https://github.com/houyuanchen111/LiNo" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/Stable-X/Hi3DGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                    </a>
              
                    </p>
                    """)
    with gr.Row():
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Input Images"):
                        input_images = gr.Gallery(
                            label="Upload Multi-light Images",
                            type="numpy",
                            columns=8, 
                            object_fit="contain",
                            preview=True,
                        )
                        input_mask = gr.Image(
                            label="Mask (Optional)",
                            type="pil",
                                height="300px",
                                 
                        )
                        run_btn = gr.Button("Run", size="lg", variant="primary")
                    
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Row(scale=3):
                        normal_output = gr.Image(label="Normal Output",height=700,)
                    gen_shape_btn = gr.Button("Generate Mesh", size="lg", variant="primary")
                    model_output = gr.Model3D(
                        label="3D Model Preview (Generated by Hi3DGen)",
                    
                    )
            
                    with gr.Row():
                        download_btn = gr.DownloadButton(
                            label="Export Mesh", 
                            interactive=False,
                            scale=1 
                        )

    
    seed = gr.Number(np.random.randint(0,1e10),visible=False)
    ss_guidance_strength =gr.Number(3,visible=False)
    ss_sampling_steps = gr.Number(50,visible=False)
    slat_guidance_strength =gr.Number(3.0,visible=False)
    slat_sampling_steps = gr.Number(6,visible=False)
    normal_bridge = gr.State()
    
    gen_shape_btn.click(
        generate_3d,
        inputs=[
            input_images, seed,  
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps,
            normal_bridge
        ],
        outputs=[model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )

    run_btn.click(
        predict_normal,
        inputs=[
            input_images,
            input_mask
        ],
        outputs=[normal_output,normal_bridge],
    )

    example_display = gr.Image(visible=False,type="pil",label="Input images")
    obj_path = gr.Textbox(label = "Name",visible=False)
    num = gr.Textbox(label = "Maximum number of images",visible=False)

    display_data = [
        [Image.open("demo/basket/demo.png"), "basket"],
        [Image.open("demo/key/demo.png"), "key"],
        [Image.open("demo/canandwood/demo.png"), "canandwood"],
        [Image.open("demo/cat/demo.png"), "cat"],
        [Image.open("demo/coins_and_keyboard/demo.png"), "coins_and_keyboard"],
        [Image.open("demo/owl/demo.png"), "owl"],
        [Image.open("demo/rabit/demo.png"), "rabit"],
        [Image.open("demo/reading/demo.png"), "reading"],
    ]
    gr.Markdown(
        """
        <p style='color: #2b93d6; font-size: 1em; text-align: left;'>
            Click any row to load an example.
        </p>
        """
    )
    gr.Examples(
        examples=display_data,
        inputs=[example_display,obj_path], 
        label="Examples"
    )
    example_display.change(
        fn=load_example_data,           
        inputs=[obj_path],   
        outputs=[                       
             input_mask,
             input_images
        ]
    )

if __name__ == "__main__":
    # Download and cache the weights
    cache_weights(WEIGHTS_DIR)
    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline.cuda()
    lino = LiNo_UniPS()
    lino.from_pretrained("weights/lino/lino.pth")
  
    demo.launch(share=False, server_name="0.0.0.0")

    # Add custom CSS for better styling
    demo_css = """
    footer {visibility: hidden}
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .upload-section {
        border: 2px dashed #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .output-section {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        background-color: #fafafa;
    }
    .button-primary {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .button-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    .examples-section {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    """

    # Update the Gradio interface with improved layout
    with gr.Blocks(css=demo_css, title="LiNo - Light of Normals") as demo:
        # Header section
        with gr.Row():
            with gr.Column():
                gr.HTML("""
                    <div class="main-header">
                        <h1 style='margin: 0; font-size: 2.5em; font-weight: 700;'>
                            Light of Normals (LiNo)
                        </h1>
                        <p style='margin: 0.5rem 0 0 0; font-size: 1.2em; opacity: 0.9;'>
                            Unified Feature Representation for Universal Photometric Stereo
                        </p>
                    </div>
                """)
        
        # Links section
        with gr.Row():
            gr.HTML("""
                <div style="text-align: center; margin: 1rem 0;">
                    <a href="https://houyuanchen111.github.io/lino.github.io/" target="_blank" style="margin: 0 10px;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg" alt="Website">
                    </a>
                    <a href="#" target="_blank" style="margin: 0 10px;">
                        <img src="https://img.shields.io/badge/Github-Page-black" alt="GitHub">
                    </a>
                    <a href="https://github.com/houyuanchen111/LiNo" target="_blank" style="margin: 0 10px;">
                        <img src="https://img.shields.io/github/stars/houyuanchen111/LiNo?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="GitHub Stars">
                    </a>
                </div>
            """)

        # Main processing section
        with gr.Row(equal_height=True):
            # Input column
            with gr.Column(scale=5, min_width=400):
                gr.Markdown("### 📸 Input Configuration", elem_classes=["section-header"])
                
                with gr.Group(elem_classes=["upload-section"]):
                    input_images = gr.Gallery(
                        label="📂 Upload Multi-light Images",
                        type="numpy",
                        columns=4,
                        rows=2,
                        object_fit="contain",
                        preview=True,
                        height=300,
                        show_label=True
                    )
                    
                    input_mask = gr.Image(
                        label="🎭 Mask (Optional)",
                        type="pil",
                        height=200,
                        show_label=True
                    )
                    
                    with gr.Row():
                        run_btn = gr.Button(
                            "🚀 Predict Normal Map", 
                            size="lg", 
                            variant="primary",
                            elem_classes=["button-primary"],
                            scale=2
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear", 
                            size="lg", 
                            variant="secondary",
                            scale=1
                        )

            # Output column
            with gr.Column(scale=7, min_width=500):
                gr.Markdown("### 🎯 Results", elem_classes=["section-header"])
                
                with gr.Group(elem_classes=["output-section"]):
                    with gr.Tabs():
                        with gr.Tab("🗺️ Normal Map"):
                            normal_output = gr.Image(
                                label="Generated Normal Map",
                                height=400,
                                show_label=False,
                                interactive=False
                            )
                            
                            with gr.Row():
                                gen_shape_btn = gr.Button(
                                    "🎲 Generate 3D Mesh", 
                                    size="lg", 
                                    variant="primary",
                                    elem_classes=["button-primary"],
                                    interactive=False
                                )
                        
                        with gr.Tab("🎯 3D Model"):
                            model_output = gr.Model3D(
                                label="3D Model Preview",
                                height=500,
                                show_label=False
                            )
                            
                            with gr.Row():
                                download_btn = gr.DownloadButton(
                                    label="📥 Download Mesh (.glb)", 
                                    interactive=False,
                                    variant="primary"
                                )

        # Advanced settings (collapsible)
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**3D Generation Settings**")
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        minimum=-1,
                        maximum=MAX_SEED
                    )
                    ss_guidance_strength = gr.Slider(
                        label="Structure Guidance",
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=0.1
                    )
                    ss_sampling_steps = gr.Slider(
                        label="Structure Steps",
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1
                    )
                
                with gr.Column():
                    gr.Markdown("**Mesh Refinement Settings**")
                    slat_guidance_strength = gr.Slider(
                        label="Mesh Guidance",
                        minimum=1,
                        maximum=10,
                        value=3.0,
                        step=0.1
                    )
                    slat_sampling_steps = gr.Slider(
                        label="Mesh Steps",
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1
                    )

        # Examples section
        with gr.Group(elem_classes=["examples-section"]):
            gr.Markdown("### 📋 Example Gallery")
            gr.Markdown("Click any example below to load it into the interface:")
            
            example_display = gr.Image(visible=False, type="pil")
            obj_path = gr.Textbox(visible=False)
            
            gr.Examples(
                examples=display_data,
                inputs=[example_display, obj_path],
                label="",
                examples_per_page=8
            )

        # Hidden state variables
        normal_bridge = gr.State()

        # Event handlers
        def enable_generate_button():
            return gr.Button(interactive=True)
        
        def disable_generate_button():
            return gr.Button(interactive=False)
        
        def clear_all():
            return (
                None,  # input_images
                None,  # input_mask
                None,  # normal_output
                None,  # model_output
                gr.Button(interactive=False),  # gen_shape_btn
                gr.Button(interactive=False),  # download_btn
            )

        # Connect event handlers
        run_btn.click(
            predict_normal,
            inputs=[input_images, input_mask],
            outputs=[normal_output, normal_bridge],
        ).then(
            enable_generate_button,
            outputs=[gen_shape_btn]
        )

        gen_shape_btn.click(
            generate_3d,
            inputs=[
                input_images, seed,
                ss_guidance_strength, ss_sampling_steps,
                slat_guidance_strength, slat_sampling_steps,
                normal_bridge
            ],
            outputs=[model_output, download_btn]
        ).then(
            lambda: gr.Button(interactive=True),
            outputs=[download_btn]
        )

        clear_btn.click(
            clear_all,
            outputs=[
                input_images, input_mask, normal_output, 
                model_output, gen_shape_btn, download_btn
            ]
        )

        example_display.change(
            fn=load_example_data,
            inputs=[obj_path],
            outputs=[input_mask, input_images]
        )

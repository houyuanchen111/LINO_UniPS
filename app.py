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
# import spaces

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
# @spaces.GPU
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
# @spaces.GPU
def predict_normal(input_images,input_mask):
    test_dataset = DemoData(input_imgs_list=input_images,input_mask=input_mask)
    test_data = test_dataset[0]

    # test_loader = DataLoader(test_dataset, batch_size=1)

    # trainer = pl.Trainer(accelerator="auto", devices=1,precision="bf16-mixed")
    # nml_predict = trainer.predict(model=lino, dataloaders=test_loader)
    nml_predict = lino(test_data)


    nml_output = 0.5 * nml_predict[0] + 0.5
    
    return ((nml_output*255.0).astype(np.uint8), nml_predict[0])

def convert_mesh(mesh_path, export_format):
    """Download the mesh in the selected format."""
    if not mesh_path:
        return None
    
    # Create a temporary file to store the mesh data
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False)
    temp_file_path = temp_file.name
    
    new_mesh_path = mesh_path.replace(".glb", f".{export_format}")
    mesh = trimesh.load_mesh(mesh_path)
    mesh.export(temp_file_path)  # Export to the temporary file
    
    return temp_file_path # Return the path to the temporary file

def load_example_data(path,numberofimages):
    path = os.path.join("demo", path)
    mask_path = os.path.join(path,"mask.png")
    image_pathes = glob.glob(os.path.join(path, f"L*")) + glob.glob(os.path.join(path, f"0*"))
    image_pathes = image_pathes[:numberofimages]
    input_images = []
    for p in image_pathes:
        input_images.append(Image.open(p))
    
    if os.path.exists(mask_path):
        input_mask = Image.open(mask_path)
    else:
        input_mask =Image.fromarray(np.ones_like(np.array(input_images[0])))
    normal_path = os.path.join(path,"normal.png")
    if os.path.exists(normal_path):
        normal_gt = Image.open(normal_path)
    else:
        normal_gt = Image.fromarray(np.ones_like(np.array(input_images[0])))
    return input_mask,input_images,normal_gt

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
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/houyuanchen111/LINO_UniPS" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/Github-Page-black" alt="badge-github-stars">
                    </a>
              
                    </p>
                    """)
    with gr.Row():
       gr.Markdown(
        """    
        LiNo-UniPS is a method for Univeral Photometric Stereo. It predicts the normal map from a given set of images. Key features include:
        
        * **Light-Agnostic:** Does not require specific lighting parameters as input.
        * **Arbitrary-Resolution:** Supports inputs of any resolution.
        * **Mask-Free:** Also supports mask-free scene normal reconstruction.
        """
    )
    with gr.Row():
        gr.Markdown(
                """
                ### Getting Started:

                1.  **Upload Your Data**: Use the "Upload Multi-light Images" button on the left to provide your input. For best results, we recommend providing 6 or more images.
                
                2.  **Upload Your Mask (Optional)**: A mask is not required for scene reconstruction. However, to reconstruct the normal map for a specific **object**, providing a mask is highly recommended. Use the "Mask" button on the left.
                
                3.  **Reconstruct**: Click the "Run" button to start the reconstruction process. You can use the slider in "Advanced Settings" to control the number of multi-light images used by LiNo-UniPS. Note: If the selected number exceeds the total number of uploaded images, the maximum available number will be used instead.
                
                4.  **Visualize**: The result will appear in the "Normal Output" viewer on the right. If you use one of our provided examples that includes a ground truth normal map, it will be displayed in the "Ground Truth" viewer for comparison.
                
                5.  **Generate Mesh (Optional)**: After the normal map is reconstructed, you can click the "Generate Mesh" button. This will use the predicted normal as a "normal bridge" to generate the corresponding 3D mesh via Hi3DGen. We recommend this step primarily for **objects**, as Hi3DGen is currently an object-level model.
                """
            )
    with gr.Row():

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.Tab("Input Images"):
                  
                        with gr.Row():
                            input_mask = gr.Image(
                                label="Mask (Optional)",
                                type="pil",
                                 height="300px",
                                 
                            )
                        input_images = gr.Gallery(
                            label="Upload Multi-light Images",
                            type="numpy",
                            columns=8, 
                            object_fit="contain",
                            preview=True,
                        )
                
               
                model_output = gr.Model3D(
                    label="3D Model Preview (Generated by Hi3DGen)",
                   
                )
          
                with gr.Row():
                    export_format = gr.Dropdown(
                        choices=["obj", "glb", "ply", "stl"],
                        value="glb",
                        label="File Format",
                        scale=2 
                    )
                    download_btn = gr.DownloadButton(
                        label="Export Mesh", 
                        interactive=False,
                        scale=1 
                    )

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("LiNo-UniPS Output"):
                        with gr.Row(scale=3):
                            normal_output = gr.Image(label="Normal Output",height=700,)
                            normal_gt = gr.Image(label="Ground Truth",height=700)
                        with gr.Accordion("Advanced Settings", open=True):
                            numberofimages = gr.Slider(0, 100, label="Number of Images", value=16, step=1)
                     
                        run_btn = gr.Button("Run", size="lg", variant="primary")
                        gen_shape_btn = gr.Button("Generate Mesh", size="lg", variant="primary")
            
    
           
    
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
    
    def update_download_button(mesh_path, export_format):
        if not mesh_path:
            return gr.File.update(value=None, interactive=False)
        
        download_path = convert_mesh(mesh_path, export_format)
        return download_path
    
    export_format.change(
        update_download_button,
        inputs=[model_output, export_format],
        outputs=[download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )

    example_display = gr.Image(visible=False,type="pil",label="Input images")
    obj_path = gr.Textbox(label = "Name",visible=False)
    num = gr.Textbox(label = "Maximum number of images",visible=False)
    is_mask = gr.Textbox(label = "Mask",visible=False)
    is_gt =  gr.Textbox(label = "Normal ground truth",visible=False)
    image_type = gr.Textbox(label = "Image type",visible=False)
    image_resolution = gr.Textbox(label = "Image resolution",visible=False)


    display_data = [

        [Image.open("demo/basket/demo.png"), "basket", 8, False, False, "Real","960*960"],
        [Image.open("demo/key/demo.png"), "key", 8, True, False, "Real","640*640"],
        [Image.open("demo/canandwood/demo.png"), "canandwood", 18, True, False, "Real","4032*2268"],
        [Image.open("demo/cat/demo.png"), "cat", 96, True, True, "Real","512*612"],
        [Image.open("demo/coins_and_keyboard/demo.png"), "coins_and_keyboard", 12, False, False, "Real","4000*4000"],
        [Image.open("demo/owl/demo.png"), "owl", 13, True, False, "Real","2400*1600"],
        [Image.open("demo/rabit/demo.png"), "rabit", 9, True, False, "Real","4000*4000"],
        [Image.open("demo/reading/demo.png"), "reading", 96, True, True, "Real","512*612"],
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
        inputs=[example_display,obj_path,num,is_mask,is_gt,image_type,image_resolution], 
        label="Examples"
    )
    example_display.change(
        fn=load_example_data,           
        inputs=[obj_path,numberofimages],   
        outputs=[                       
             input_mask,
             input_images,
             normal_gt
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



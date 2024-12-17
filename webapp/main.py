import gradio as gr

from processsors import RootSegmentor
from processsors import *

from gradio_imageslider import ImageSlider

import cv2 as cv

PRELOAD_MODELS = False

if PRELOAD_MODELS:
    root_segmentor = RootSegmentor()
    
    
def process(input_img, model_type):
    
    print(model_type)
    
    if PRELOAD_MODELS:
        global root_segmentor
    else:
        root_segmentor = RootSegmentor(model_type)
        
    result = root_segmentor.predict(input_img)
    
    return result

def just_show(files, should_process, model_type):
    
    imgs = []
    
    img = merge_images(files)
    
    
    
    imgs.append(img)
    
    if should_process:    
        root_segmentor = RootSegmentor(model_type)
    
    results = []

    for file in files:
        print(type(file))
        print(file)
        img = cv.imread(file)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #imgs.append(img)
        
        if should_process: 
        
            result = root_segmentor.predict(img)
            results.append(result)
            #imgs.append(results)
        
    if should_process: 
        img_res = merge_images(results)
        imgs.append(img_res)

    return imgs

def slider_test(img1, img2):
 
    return [img1,img2]

def download_result():
    
    #print(filepath)
    return
    

def gui():

  with gr.Blocks(title="Root analysis", theme=gr.themes.Soft()) as demo:

    big_block = gr.HTML("""

    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: white
            margin: 0;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px;
            color: #fff;
        }

        hr {
            border: 1px solid #ddd;
            margin: 5px;
        }

    </style>

    <header>
        <div style="display: flex; align-items: center;">
            <div style="text-align: left;">
            <h1>Root Analysis</h1>
            <p>Root segmentation using underground root scanner images.</p>
            <h3>Tropical Forages Program</h3>
            <p><b>Authors: </b>Andres Felipe Ruiz-Hurtado, Juan Andrés Cardoso Arango</p>
            <p></p>            
        </div>
        </div>
        <div style="background-color: white; padding: 5px; border-radius: 15px; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);">
                        <img src="https://alliancebioversityciat.org/sites/default/files/styles/1920_scale/public/images/Alliance%20Logo%20Refresh-color.jpg" alt="Logo" width="200" height="100">
                    </div>
    </header>   
    
    """)
    
    #<iframe style="height:600px;width: 100%;" src="/file=slides.html" title="description"></iframe>

    
    #<iframe style="height:600px;width: 100%;" src="https://revealjs.com/demo/?view" title="description"></iframe>

    with gr.Tab("Single Image"):
    
        model_selector = gr.Dropdown(
                ["segroot_finetuned", "segroot", "segroot_finetuned_dec", "seg_model"], label="Model"
                , info="AI model"
                ,value="segroot_finetuned"
            )

        input_img=gr.Image(render=False)
        output_img=gr.Image(render=False)

        gr.Interface(
            fn=process,
            inputs=[input_img,model_selector],
            outputs=output_img,
            examples=[["example_1.jpg"],["example_2.jpg"],["example_3.jpg"]]
        )

        #examples = gr.Examples([["Chicago"], ["Little Rock"], ["San Francisco"]], textbox)

        with gr.Row():
            img_comp = ImageSlider(label="Root Segmentation")
        with gr.Row():
            compare_button = gr.Button("Compare")
            compare_button.click(fn=slider_test, inputs=[input_img,output_img], outputs=img_comp, api_name="slider_test")
        
    with gr.Tab("Multiple Images"):

    #img_comp = ImageSlider(label="Blur image", type="pil")
    
        gallery = gr.Gallery(show_fullscreen_button=True, render=False) 
        
        gr.Interface(
            fn=just_show
            ,inputs=[gr.File(file_count="multiple"),gr.Checkbox(label="Process", info="Check if you want to process"),model_selector]
            ,outputs= gallery
            , examples=[[["example_1.jpg", "example_2.jpg", "example_3.jpg"]]]
        )

    with gr.Tab("Compare"):

        img_comp = ImageSlider(label="Root Segmentation")
        img_comp.upload(inputs=img_comp, outputs=img_comp)

    
    #d = gr.DownloadButton("Download the file")
    #d.click(download_result, gallery, None)
    
    # with gr.Row():    
    #     img1=gr.Image()
    #     img2=gr.Image()
    # with gr.Row():
    #     img_comp = ImageSlider(label="Blur image", type="pil")
    # with gr.Row():
    #     compare_button = gr.Button("Compare")
    #     compare_button.click(fn=slider_test, inputs=[img1,img2], outputs=img_comp, api_name="slider_test")
    
    # with gr.Group():
    #     img_comp = ImageSlider(label="Blur image", type="pil")
    #     #img1.upload(slider_test, inputs=[img1,img2], outputs=img_comp)
    #     gr.Interface(slider_test, inputs=[img1,img2], outputs=img_comp)
    
    demo.launch(allowed_paths=["logo.png"], share=False)

if __name__ == "__main__":
    gui()
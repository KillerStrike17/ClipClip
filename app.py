import gradio as gr
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, DetrFeatureExtractor, DetrForObjectDetection, AutoFeatureExtractor, AutoModelForObjectDetection
import torch

feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
dmodel = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_image(image, text, prob, num=1):
    
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = dmodel(**inputs)
    
    # model predicts bounding boxes and corresponding COCO classes
    logits = outputs.logits
    bboxes = outputs.pred_boxes
    probas = outputs.logits.softmax(-1)[0, :, :-1] #removing no class as detr maps 
    
    keep = probas.max(-1).values > prob
    outs = feature_extractor.post_process(outputs, torch.tensor(image.size[::-1]).unsqueeze(0))
    bboxes_scaled = outs[0]['boxes'][keep].detach().numpy()
    labels = outs[0]['labels'][keep].detach().numpy()
    scores = outs[0]['scores'][keep].detach().numpy()
    
    images_list = []
    for i,j in enumerate(bboxes_scaled):
      
      xmin = int(j[0])
      ymin = int(j[1])
      xmax = int(j[2])
      ymax = int(j[3])
    
      im_arr = np.array(image)
      roi = im_arr[ymin:ymax, xmin:xmax]
      roi_im = Image.fromarray(roi)
    
      images_list.append(roi_im)
    
    inpu = processor(text = [text], images=images_list , return_tensors="pt", padding=True)
    output = model(**inpu)
    logits_per_image = output.logits_per_text
    # print("Logits:",logits_per_image)
    probs = logits_per_image.softmax(-1)
    # print("Probability:",probs)
    l_idx = np.argsort(probs[-1].detach().numpy())[::-1][0:num]
    # print("Index:",l_idx)
    
    final_ims = []
    for i,j in enumerate(images_list):
      json_dict = {}
      if i in l_idx:
        json_dict['image'] = images_list[i]
        json_dict['score'] = probs[-1].detach().numpy()[i]
    
        final_ims.append(json_dict)
    
    fi = sorted(final_ims, key=lambda item: item.get("score"), reverse=True)
    return fi[0]['image'], fi[0]['score']
def zero_shot_classification(image, labels):
  labels = labels.split(',')
  text = [f"a photo of a {c}" for c in labels]
  inpu = processor(text = text, images=image , return_tensors="pt", padding=True)
  output = model(**inpu)
  logits_per_image = output.logits_per_image 
  probs = logits_per_image.softmax(dim=1)
  return {k: float(v) for k, v in zip(labels, probs[0])}

with gr.Blocks() as demo:
  with gr.Tab("Clip and Crop"):
    i1 = gr.Image(type="pil", label="Input image")
    i2 = gr.Textbox(label="Input text")
    i3 = gr.Number(default=0.96, label="Threshold percentage score")
    o1 = gr.Image(type="pil", label="Cropped part")
    o2 = gr.Textbox(label="Similarity score")
    title = "Cliping and Cropping"
    description = "<p style= 'color:white'>Extract sections of images from your image by using OpenAI's CLIP and Facebooks Detr implemented on HuggingFace Transformers, if the similarity score is not so much, then please consider the prediction to be void.</p>" 
    examples=[['ex3.jpg', 'black bag', 0.96],['ex2.jpg', 'man in red dress', 0.85]]
    article = "<p style= 'color:white; text-align:center;'><a href='https://github.com/Vishnunkumar/clipcrop' target='_blank'>clipcrop</a></p>"
    examples=[['Assets/ex3.jpg', 'black bag', 0.96],['Assets/ex2.jpg', 'man in red dress', 0.85]]
    gr.Interface(fn=extract_image, inputs=[i1, i2, i3], outputs=[o1, o2], title=title, description=description, article=article, examples=examples, enable_queue=True)
  with gr.Tab("Zero Shot Image Classification"):
    i1 = gr.Image(label="Image to classify.", type="pil")
    i2 = gr.Textbox(lines=1, label="Comma separated classes", placeholder="Enter your classes separated by ','",)
    title = "Zero Shot Image Classification"
    description = "<p style= 'color:white'>Use clip models embedding to identify the closest class it belongs form its pretrianed data from the given list</p>" 
    examples=[['Assets/cat.jpg', 'cat,dog,man'],['Assets/dog.jpg', 'cat,dog,man']]
    gr.Interface(fn=zero_shot_classification,inputs=[i1,i2],outputs="label",title=title,description="Zero Shot Image classification..", examples = examples)
demo.launch(debug = False)
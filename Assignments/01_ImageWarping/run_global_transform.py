import gradio as gr
import cv2
import numpy as np
import math

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    new_image = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)

    # boost scale
    image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)      

    # rotation and translation
    center = (image_new.shape[0] // 2 - translation_y  // 10 * 3, image_new.shape[1] // 2 + translation_x  // 10 * 3)  # 旋转中心
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            x = round(math.cos(rotation/180*math.pi)*(i - image.shape[0] // 2) + math.sin(rotation/180*math.pi)*(j - image.shape[1] // 2))
            y = round(math.sin(-rotation/180*math.pi)*(i - image.shape[0] // 2) + math.cos(rotation/180*math.pi)*(j - image.shape[1] // 2))
            image_new[x+center[0], y+center[1]] = image[i, j]

    # image_new[size_x+translation_y:size_x+image.shape[0]+translation_y, size_y+translation_x:size_y+image.shape[1]+translation_x] = image

    # flip_horizontal
    if flip_horizontal:
        for i in range(image_new.shape[0]):
            for j in range(image_new.shape[1]):
                new_image[image_new.shape[0] - 1 -i, image_new.shape[1] - 1 -j] = image_new[i,j]
        image = np.array(new_image)
    else:
        image = np.array(image_new)
    
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()

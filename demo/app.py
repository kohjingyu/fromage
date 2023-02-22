import tempfile
from share_btn import community_icon_html, loading_icon_html, share_js, save_js
import huggingface_hub
import gradio as gr
from fromage import utils
from fromage import models
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "False"


css = """
    #chatbot { min-height: 300px; }
    #save-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #save-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    #share-btn {
        background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
    }
    #share-btn:hover {
        background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
    }
    #gallery { z-index: 999999; }
    #gallery img:hover {transform: scale(2.3); z-index: 999999; position: relative; padding-right: 30%; padding-bottom: 30%;}
    #gallery button img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; padding-bottom: 0;}
    @media (hover: none) {
        #gallery img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; 0;}
    }
"""

examples = [
    'examples/sparrow.png',
    'examples/beaver.png',
    'examples/couch.png',
    'examples/guac.png',
    'examples/scraped_knee.png'
]

# Download model from HF Hub.
ckpt_path = huggingface_hub.hf_hub_download(
    repo_id='jykoh/fromage', filename='pretrained_ckpt.pth.tar')
args_path = huggingface_hub.hf_hub_download(
    repo_id='jykoh/fromage', filename='model_args.json')
model = models.load_fromage('./', args_path, ckpt_path)


def upload_image(state, image_input):
    conversation = state[0]
    chat_history = state[1]
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    input_image.save(image_input.name)  # Overwrite with smaller image.
    conversation += [(f'<img src="/file={image_input.name}" style="display: inline-block;">', "")]
    return [conversation, chat_history + [input_image, ""]], conversation


def reset():
    return [[], []], []


def reset_last(state):
    conversation = state[0][:-1]
    chat_history = state[1][:-2]
    return [conversation, chat_history], conversation


def save_image_to_local(image: Image.Image):
    # TODO(jykoh): Update so the url path is used, to prevent repeat saving.
    filename = next(tempfile._get_candidate_names()) + '.png'
    image.save(filename)
    return filename


def generate_for_prompt(input_text, state, ret_scale_factor, max_num_rets, num_words, temperature):
    # Ignore empty inputs.
    if len(input_text) == 0:
        return state, state[0], gr.update(visible=True)

    input_prompt = 'Q: ' + input_text + '\nA:'
    conversation = state[0]
    chat_history = state[1]
    print('Generating for', chat_history, flush=True)

    # If an image was uploaded, prepend it to the model.
    model_inputs = chat_history
    model_inputs.append(input_prompt)

    top_p = 1.0
    if temperature != 0.0:
        top_p = 0.95

    print('Running model.generate_for_images_and_texts with',
          model_inputs, flush=True)
    model_outputs = model.generate_for_images_and_texts(model_inputs,
                                                        num_words=max(num_words, 1), ret_scale_factor=ret_scale_factor, top_p=top_p,
                                                        temperature=temperature, max_num_rets=max_num_rets)
    print('model_outputs', model_outputs, flush=True)

    im_names = []
    response = ''
    text_outputs = []
    for output_i, output in enumerate(model_outputs):
        if type(output) == str:
            if output_i > 0:
                response += '<br/>'
            text_outputs.append(output)
            response += output
            if len(model_outputs) > 1:
                response += '<br/>'
        elif type(output) == list:
            for image in output:
                filename = save_image_to_local(image)
                response += f'<img src="/file={filename}" style="display: inline-block;">'
        elif type(output) == Image.Image:
            filename = save_image_to_local(output)
            response += f'<img src="/file={filename}" style="display: inline-block;">'

    chat_history = model_inputs + \
        [' '.join([s for s in model_outputs if type(s) == str]) + '\n']
    # Remove [RET] from outputs.
    conversation.append((input_text, response.replace('[RET]', '')))

    # Set input image to None.
    print('state', state, flush=True)
    print('updated state', [conversation, chat_history], flush=True)
    return [conversation, chat_history], conversation, gr.update(visible=True), gr.update(visible=True)


with gr.Blocks(css=css) as demo:
    gr.HTML("""
        <h1>🧀 FROMAGe</h1>
        <p>This is the official Gradio demo for the FROMAGe model, a model that can process arbitrarily interleaved image and text inputs, and produce image and text outputs.</p>

        <strong>Paper:</strong> <a href="https://arxiv.org/abs/2301.13823" target="_blank">Grounding Language Models to Images for Multimodal Generation</a>
        <br/>
        <strong>Project Website:</strong> <a href="https://jykoh.com/fromage" target="_blank">FROMAGe Website</a>
        <br/>
        <strong>Code and Models:</strong> <a href="https://github.com/kohjingyu/fromage" target="_blank">GitHub</a>
        <br/>
        <br/>

        <strong>Tips:</strong>
        <ul>
        <li>Start by inputting either image or text prompts (or both) and chat with FROMAGe to get image-and-text replies.</li>
        <li>Tweak the level of sensitivity to images and text using the parameters on the right.</li>
        <li>Check out cool conversations in the examples or community tab for inspiration and share your own!</li>
        <li>For faster inference without waiting in queue, you may duplicate the space and use your own GPU: <a href="https://huggingface.co/spaces/jykoh/fromage?duplicate=true"><img style="display: inline-block; margin-top: 0em; margin-bottom: 0em" src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></li>
        </ul>
    """)

    gr_state = gr.State([[], []])  # conversation, chat_history

    with gr.Row():
        with gr.Column(scale=0.7, min_width=500):
            with gr.Row():
                chatbot = gr.Chatbot(elem_id="chatbot", label="🧀 FROMAGe Chatbot")
            with gr.Row():
                image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])

                text_input = gr.Textbox(label="Message", placeholder="Type a message")

                with gr.Column():
                    submit_btn = gr.Button(
                        "Submit", interactive=True, variant="primary")
                    clear_last_btn = gr.Button("Undo")
                    clear_btn = gr.Button("Reset All")
                    with gr.Row(visible=False) as save_group:
                        save_button = gr.Button("💾 Save Conversation as .png", elem_id="save-btn")

                    with gr.Row(visible=False) as share_group:
                        share_button = gr.Button("🤗 Share to Community (opens new window)", elem_id="share-btn")

        with gr.Column(scale=0.3, min_width=400):
            ret_scale_factor = gr.Slider(minimum=0.0, maximum=3.0, value=1.0, step=0.1, interactive=True,
                                         label="Frequency multiplier for returning images (higher means more frequent)")
            max_ret_images = gr.Number(
                minimum=0, maximum=3, value=2, precision=1, interactive=True, label="Max images to return")
            gr_max_len = gr.Slider(minimum=1, maximum=64, value=32,
                                   step=1, interactive=True, label="Max # of words")
            gr_temperature = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.0, interactive=True, label="Temperature (0 for deterministic, higher for more randomness)")

            gallery = gr.Gallery(
                value=[Image.open(e) for e in examples], label="Example Conversations", show_label=True, elem_id="gallery",
            ).style(grid=[2], height="auto")

    text_input.submit(generate_for_prompt, [text_input, gr_state, ret_scale_factor,
                      max_ret_images, gr_max_len, gr_temperature], [gr_state, chatbot, share_group, save_group])
    text_input.submit(lambda: "", None, text_input)  # Reset chatbox.
    submit_btn.click(generate_for_prompt, [text_input, gr_state, ret_scale_factor,
                     max_ret_images, gr_max_len, gr_temperature], [gr_state, chatbot, share_group, save_group])
    submit_btn.click(lambda: "", None, text_input)  # Reset chatbox.

    image_btn.upload(upload_image, [gr_state, image_btn], [gr_state, chatbot])
    clear_last_btn.click(reset_last, [gr_state], [gr_state, chatbot])
    clear_btn.click(reset, [], [gr_state, chatbot])
    share_button.click(None, [], [], _js=share_js)
    save_button.click(None, [], [], _js=save_js)


demo.queue(concurrency_count=1, api_open=False, max_size=16)
demo.launch(debug=True, server_name="0.0.0.0")
# demo.launch(debug=True, server_name="127.0.0.1")

import gradio as gr
import numpy as np

import spaces
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers import FluxTransformer2DModel
from diffusers.utils import load_image

from huggingface_hub import hf_hub_download

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
pipe.load_lora_weights("kontext-community/relighting-kontext-dev-lora-v3", weight_name="relighting-kontext-dev-lora-v3.safetensors", adapter_name="lora")
pipe.set_adapters(["lora"], adapter_weights=[0.75])

MAX_SEED = np.iinfo(np.int32).max

# Illumination options mapping
ILLUMINATION_OPTIONS = {
# Natural Daylight
    "natural lighting": "Neutral white color temperature with balanced exposure and soft shadows",
    "sunshine from window": "Bright directional sunlight with hard shadows and visible light rays",
    "golden time": "Warm golden hour lighting with enhanced warm colors and soft shadows",
    "sunrise in the mountains": "Warm backlighting with atmospheric haze and lens flare",
    "afternoon light filtering through trees": "Dappled sunlight patterns with green color cast from foliage",
    "early morning rays, forest clearing": "God rays through trees with warm color temperature",
    "golden sunlight streaming through trees": "Golden god rays with atmospheric particles in light beams",
    
    # Sunset & Evening
    "sunset over sea": "Warm sunset light with soft diffused lighting and gentle gradients",
    "golden hour in a meadow": "Golden backlighting with lens flare and rim lighting",
    "golden hour on a city skyline": "Golden lighting on buildings with silhouette effects",
    "evening glow in the desert": "Warm directional lighting with long shadows",
    "dusky evening on a beach": "Cool backlighting with horizon silhouettes",
    "mellow evening glow on a lake": "Warm lighting with water reflections",
    "warm sunset in a rural village": "Golden hour lighting with peaceful warm tones",
    
    # Night & Moonlight
    "moonlight through curtains": "Cool blue lighting with curtain shadow patterns",
    "moonlight in a dark alley": "Cool blue lighting with deep urban shadows",
    "midnight in the forest": "Very low brightness with minimal ambient lighting",
    "midnight sky with bright starlight": "Cool blue lighting with star point sources",
    "fireflies lighting up a summer night": "Small glowing points with warm ambient lighting",
    
    # Indoor & Cozy
    "warm atmosphere, at home, bedroom": "Very warm lighting with soft diffused glow",
    "home atmosphere, cozy bedroom illumination": "Warm table lamp lighting with pools of light",
    "cozy candlelight": "Warm orange flickering light with dramatic shadows",
    "candle-lit room, rustic vibe": "Multiple warm candlelight sources with atmospheric shadows",
    "night, cozy warm light from fireplace": "Warm orange-red firelight with flickering effects",
    "campfire light": "Warm orange flickering light from below with dancing shadows",
    
    # Urban & Neon
    "neon night, city": "Vibrant blue, magenta, and green neon lights with reflections",
    "blue neon light, urban street": "Blue neon lighting with urban glow effects",
    "neon, Wong Kar-wai, warm": "Warm amber and red neon with moody selective lighting",
    "red and blue police lights in rain": "Alternating red and blue strobing with wet reflections",
    "red glow, emergency lights": "Red emergency lighting with harsh shadows and high contrast",
    
    # Sci-Fi & Fantasy
    "sci-fi RGB glowing, cyberpunk": "Electric blue, pink, and green RGB lighting with glowing effects",
    "rainbow reflections, neon": "Chromatic rainbow patterns with prismatic reflections",
    "magic lit": "Colored rim lighting in purple and blue with soft ethereal glow",
    "mystical glow, enchanted forest": "Supernatural green and blue glowing with floating particles",
    "ethereal glow, magical forest": "Supernatural lighting with blue-green rim lighting",
    "underwater glow, deep sea": "Blue-green lighting with caustic patterns and particles",
    "underwater luminescence": "Blue-green bioluminescent glow with caustic light patterns",
    "aurora borealis glow, arctic landscape": "Green and purple dancing sky lighting",
    "crystal reflections in a cave": "Sparkle effects with prismatic light dispersion",
    
    # Weather & Atmosphere
    "foggy forest at dawn": "Volumetric fog with cool god rays through trees",
    "foggy morning, muted light": "Soft fog effects with reduced contrast throughout",
    "soft, diffused foggy glow": "Heavy fog with soft lighting and no harsh shadows",
    "stormy sky lighting": "Dramatic lighting with high contrast and rim lighting",
    "lightning flash in storm": "Brief intense white light with stark shadows",
    "rain-soaked reflections in city lights": "Wet surface reflections with streaking light effects",
    "gentle snowfall at dusk": "Cool blue lighting with snowflake particle effects",
    "hazy light of a winter morning": "Neutral lighting with atmospheric haze",
    "mysterious twilight, heavy mist": "Heavy fog with cool lighting and atmospheric depth",
    
    # Seasonal & Nature
    "vibrant autumn lighting in a forest": "Enhanced warm autumn colors with dappled sunlight",
    "purple and pink hues at twilight": "Warm lighting with soft purple and pink color grading",
    "desert sunset with mirage-like glow": "Warm orange lighting with heat distortion effects",
    "sunrise through foggy mountains": "Warm lighting through mist with atmospheric perspective",
    
    # Professional & Studio
    "soft studio lighting": "Multiple diffused sources with even illumination and minimal shadows",
    "harsh, industrial lighting": "Bright fluorescent lighting with hard shadows",
    "fluorescent office lighting": "Cool white overhead lighting with slight green tint",
    "harsh spotlight in dark room": "Single intense directional light with dramatic shadows",
    
    # Special Effects & Drama
    "light and shadow": "Maximum contrast with sharp shadow boundaries",
    "shadow from window": "Window frame shadow patterns with geometric shapes",
    "apocalyptic, smoky atmosphere": "Orange-red fire tint with smoke effects",
    "evil, gothic, in a cave": "Low brightness with cool lighting and deep shadows",
    "flickering light in a haunted house": "Unstable flickering with cool and warm mixed lighting",
    "golden beams piercing through storm clouds": "Dramatic god rays with high contrast",
    "dim candlelight in a gothic castle": "Warm orange candlelight with stone texture enhancement",
    
    # Festival & Celebration
    "colorful lantern light at festival": "Multiple colored lantern sources with bokeh effects",
    "golden glow at a fairground": "Warm carnival lighting with colorful bulb effects",
    "soft glow through stained glass": "Colored light filtering with rainbow surface patterns",
    "glowing embers from a forge": "Orange-red glowing particles with intense heat effects"

    }

# Lighting direction options
DIRECTION_OPTIONS = {
    "auto": "",  
    "left side": "Position the light source from the left side of the frame, creating shadows falling to the right.",
    "right side": "Position the light source from the right side of the frame, creating shadows falling to the left.",
    "top": "Position the light source from directly above, creating downward shadows.",
    "top left": "Position the light source from the top left corner, creating diagonal shadows falling down and to the right.",
    "top right": "Position the light source from the top right corner, creating diagonal shadows falling down and to the left.",
    "bottom": "Position the light source from below, creating upward shadows and dramatic under-lighting.",
    "front": "Position the light source from the front, minimizing shadows and creating even illumination.",
    "back": "Position the light source from behind the subject, creating silhouette effects and rim lighting."
}

@spaces.GPU
def infer(input_image, prompt, illumination_dropdown, direction_dropdown, seed=42, randomize_seed=False, guidance_scale=2.5, progress=gr.Progress(track_tqdm=True)):
    """
    Performs relighting on an input image using the FLUX.1-Kontext model.
    
    This function constructs a detailed prompt based on user selections from dropdowns
    or a custom text prompt. It then uses the diffusers pipeline to generate a new
    image with the specified lighting, while aiming to preserve the original subject.
    
    The logic prioritizes the custom text in the 'prompt' box. If the user types
    their own prompt, it will be used directly, overriding the 'illumination_dropdown'
    selection. The dropdowns serve as convenient presets to populate the prompt box.
    
    Args:
        input_image (PIL.Image.Image): The input image to be relighted.
        prompt (str): The detailed text description of the desired lighting effect.
                      If this is manually filled, it overrides the illumination dropdown.
        illumination_dropdown (str): A preset lighting style. See "Dropdown Options" below.
        direction_dropdown (str): A preset for the light's direction. See "Dropdown Options" below.
        seed (int): The seed for the random number generator for reproducibility.
        randomize_seed (bool): If True, a random seed is used, overriding the 'seed' value.
        guidance_scale (float): Controls how closely the model follows the prompt.
        progress (gr.Progress): A Gradio progress tracker for the UI.
    
    Returns:
        tuple[list[PIL.Image.Image], int, str]: A tuple containing:
            - A list with the input image (image[0]) and the relighted output image (image[1]).
            - The seed used for the generation.
            - The final constructed prompt string used by the model.
    
    ------------------------------------------------------------------------------------
    Dropdown Options
    ------------------------------------------------------------------------------------
    
    **illumination_dropdown Options:**
    
      - "natural lighting"
      - "sunshine from window"
      - "golden time"
      - "sunrise in the mountains"
      - "afternoon light filtering through trees"
      - "early morning rays, forest clearing"
      - "golden sunlight streaming through trees"
      - "sunset over sea"
      - "golden hour in a meadow"
      - "golden hour on a city skyline"
      - "evening glow in the desert"
      - "dusky evening on a beach"
      - "mellow evening glow on a lake"
      - "warm sunset in a rural village"
      - "moonlight through curtains"
      - "moonlight in a dark alley"
      - "midnight in the forest"
      - "midnight sky with bright starlight"
      - "fireflies lighting up a summer night"
      - "warm atmosphere, at home, bedroom"
      - "home atmosphere, cozy bedroom illumination"
      - "cozy candlelight"
      - "candle-lit room, rustic vibe"
      - "night, cozy warm light from fireplace"
      - "campfire light"
      - "neon night, city"
      - "blue neon light, urban street"
      - "neon, Wong Kar-wai, warm"
      - "red and blue police lights in rain"
      - "red glow, emergency lights"
      - "sci-fi RGB glowing, cyberpunk"
      - "rainbow reflections, neon"
      - "magic lit"
      - "mystical glow, enchanted forest"
      - "ethereal glow, magical forest"
      - "underwater glow, deep sea"
      - "underwater luminescence"
      - "aurora borealis glow, arctic landscape"
      - "crystal reflections in a cave"
      - "foggy forest at dawn"
      - "foggy morning, muted light"
      - "soft, diffused foggy glow"
      - "stormy sky lighting"
      - "lightning flash in storm"
      - "rain-soaked reflections in city lights"
      - "gentle snowfall at dusk"
      - "hazy light of a winter morning"
      - "mysterious twilight, heavy mist"
      - "vibrant autumn lighting in a forest"
      - "purple and pink hues at twilight"
      - "desert sunset with mirage-like glow"
      - "sunrise through foggy mountains"
      - "soft studio lighting"
      - "harsh, industrial lighting"
      - "fluorescent office lighting"
      - "harsh spotlight in dark room"
      - "light and shadow"
      - "shadow from window"
      - "apocalyptic, smoky atmosphere"
      - "evil, gothic, in a cave"
      - "flickering light in a haunted house"
      - "golden beams piercing through storm clouds"
      - "dim candlelight in a gothic castle"
      - "colorful lantern light at festival"
      - "golden glow at a fairground"
      - "soft glow through stained glass"
      - "glowing embers from a forge"
      - "custom"
    
    **direction_dropdown Options:**
    - "auto"
    - "left side"
    - "right side"
    - "top"
    - "top left"
    - "top right"
    - "bottom"
    - "front"
    - "back"
    
    ------------------------------------------------------------------------------------
    How to Write Custom Prompts
    ------------------------------------------------------------------------------------
    
    For full creative control, write your own prompt in the "Prompt" textbox. This will
    override any selection from the "Choose Lighting Style" dropdown.
    
    **1. Be Descriptive and Specific:**
    - *Weak Prompt:* `candlelight`
    - *Strong Prompt:* `Warm orange flickering light from a single candle, casting dramatic,
      dancing shadows on the wall.`
    
    **2. Focus on the Characteristics of Light:**
    Build your prompt by considering these key elements:
    - *Color Temperature:* Is the light warm, cool, golden, blue, orange-red, or neutral white?
    - *Quality & Hardness:* Is it soft and diffused (gentle shadows) or hard and direct
      (sharp, high-contrast shadows)?
    - *Source:* Where is the light coming from? A window, a fireplace, a neon sign, the moon?
    - *Atmospheric Effects:* Do you want `god rays`, `lens flare`, `volumetric fog`, `bokeh`,
      `dust particles in light beams`, or `smoke`?
    - *Shadows:* Describe them. Are they `long`, `short`, `deep`, `soft`, or `sharp-edged`?
    """
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
        
    input_image = input_image.convert("RGB")

    #If the dropdown isn't custom, and the user didn't specify a prompt, fill the prompt with the correct one from the illumination options
    if illumination_dropdown != "custom" and prompt == "":
        prompt = ILLUMINATION_OPTIONS[illumination_dropdown]

    #If the prompt matches the illumination options, prefix that
    if illumination_dropdown != "custom" and prompt == ILLUMINATION_OPTIONS[illumination_dropdown]:
        prompt_prefix = f", with {illumination_dropdown}"
    #If the prompt was changed, the prefix is empty as the user prompt is predominant
    else:
        prompt_prefix = ""

    # If direction isn't auto, add the direction suffix
    if direction_dropdown != "auto" and prompt_prefix != "":
        prompt_prefix = prompt_prefix + f"coming from the {direction_dropdown} of the image"
    elif direction_dropdown != "auto" and prompt_prefix == "":
        prompt_prefix = f", light coming from the {direction_dropdown} of the image"
    
    prompt_with_template = f"Relight the image{prompt_prefix}. {prompt} Maintain the identity of the foreground subjects."
    
    print(prompt_with_template)
    
    image = pipe(
        image=input_image, 
        prompt=prompt_with_template,
        guidance_scale=guidance_scale,
        width=input_image.size[0],
        height=input_image.size[1],
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    return [input_image, image], seed, prompt_with_template

def update_prompt_from_dropdown(illumination_option):
    """Update the prompt textbox based on dropdown selection"""
    if illumination_option == "custom":
        return ""  # Clear the prompt for custom input
    else:
        return ILLUMINATION_OPTIONS[illumination_option]

css="""
#col-container {
    margin: 0 auto;
    max-width: 1020px;
}
"""

with gr.Blocks(css=css) as demo:
    
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# Kontext Relight 💡
        """)
        gr.Markdown(f"""Flux Kontext[dev] finetuned for scene relighting ✨ [Download the LoRA](https://huggingface.co/kontext-community/relighting-kontext-dev-lora-v3/blob/main/relighting-kontext-dev-lora-v3.safetensors)
         """)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload the image for relighting", type="pil")
                
                with gr.Row():
                    
                    illumination_dropdown = gr.Dropdown(
                        choices=["custom"] + list(ILLUMINATION_OPTIONS.keys()),
                        value="sunshine from window",
                        label="Choose Lighting Style",
                        scale=2
                    )
                    
                    direction_dropdown = gr.Dropdown(
                        choices=list(DIRECTION_OPTIONS.keys()),
                        value="auto",
                        label="Light Direction",
                        scale=1
                    )
                prompt = gr.Textbox(
                    label="Prompt",
                    info="Prompt is generated by the selected illumination style, but you can override it by writing your own",
                    show_label=True,
                    max_lines=3,
                    placeholder="select an illumination style above or type your custom description...",
                    value="Add directional sunlight from window source. Increase brightness on lit areas. Create hard shadows with sharp edges. Set warm white color temperature. Add visible light rays and dust particles in beams.",
                    container=True
                )
                
                
                run_button = gr.Button("Run", scale=0, variant="primary")
                
                with gr.Accordion("Advanced Settings", open=False):
            
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0,
                    )
                    
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1,
                        maximum=10,
                        step=0.1,
                        value=2.5,
                    )       
                    
            with gr.Column():
                result = gr.ImageSlider(label="Result", show_label=False, interactive=False)
                final_prompt = gr.Textbox(label="Processed prompt", info="The structure of prompt to use if you download the LoRA")
        # update prompt when dropdown changes
        illumination_dropdown.change(
            fn=update_prompt_from_dropdown,
            inputs=[illumination_dropdown],
            outputs=[prompt]
        )
        
        gr.Examples(
            examples=[
                ["./assets/pexels-creationhill-1681010.jpg", "Add multiple colored light sources from lanterns. Create warm festival lighting. Set varied color temperatures. Add bokeh effects.", "colorful lantern light at festival", "auto", 0, True, 2.5],
                ["./assets/pexels-creationhill-1681010.jpg",  "add futuristic RGB lighting with electric blues, hot pinks, and neon greens creating a high-tech atmosphere with dramatic color separation and glowing effects", "sci-fi RGB glowing, cyberpunk", "left side", 0, True, 2.5],
                ["./assets/pexels-moose-photos-170195-1587009.jpg",  "Set blue-green color temperature. Add volumetric lighting effects. Reduce red channel significantly. Create particle effects in light beams. Add caustic light patterns.", "underwater glow, deep sea", "top", 0, True, 2.5],
                ["./assets/pexels-moose-photos-170195-1587009.jpg", "Replace lighting with red sources. Add flashing strobing effects. Increase contrast. Create harsh shadows. Set monochromatic red color scheme.", "red glow, emergency lights", "right side", 0, True, 2.5],
                ["./assets/pexels-simon-robben-55958-614810.jpg",  "Add directional sunlight from window source. Increase brightness on lit areas. Create hard shadows with sharp edges. Set warm white color temperature. Add visible light rays and dust particles in beams.", "sunshine from window", "top right", 0, True, 2.5],
                ["./assets/pexels-simon-robben-55958-614810.jpg", "add vibrant neon lights in electric blues, magentas, and greens casting colorful reflections on surfaces, creating a cyberpunk urban atmosphere with dramatic color contrasts", "neon night, city", "top left", 0, True, 2.5],
                ["./assets/pexels-freestockpro-1227513.jpg",  "warm lighting with soft purple and pink color grading", "purple and pink hues at twilight", "auto", 0, True, 2.5],
                ["./assets/pexels-pixabay-158827.jpg", "Soft fog effects with reduced contrast throughout", "foggy morning, muted light", "auto", 0, True, 2.5],
                ["./assets/pexels-pixabay-355465.jpg", "daylight, bright sunshine", "custom", "auto", 0, True, 2.5]           
            ],
            inputs=[input_image, prompt, illumination_dropdown, direction_dropdown, seed, randomize_seed, guidance_scale],
            outputs=[result, seed, final_prompt],
            fn=infer,
            cache_examples="lazy"
        )
    
        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn = infer,
            inputs = [input_image, prompt, illumination_dropdown, direction_dropdown, seed, randomize_seed, guidance_scale],
            outputs = [result, seed, final_prompt]
        )
    

demo.launch(mcp_server=True)
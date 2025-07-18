import torch
import random
from PIL import Image
from diffusers_fixed.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
from diffusers_fixed.models.flux_transformer_2d import FluxTransformer2DModel
from diffusers_fixed.utils import load_image
from replicate import Predictor

class Predict(Predictor):
    def setup(self):
        self.pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.load_lora_weights("kontext-community/relighting-kontext-dev-lora-v3", weight_name="relighting-kontext-dev-lora-v3.safetensors", adapter_name="lora")
        self.pipe.set_adapters(["lora"], adapter_weights=[0.75])

    def predict(self, image, prompt, illumination, direction, seed, randomize_seed, guidance_scale):
        return infer(image, prompt, illumination, direction, seed, randomize_seed, guidance_scale)

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

def infer(input_image, prompt, illumination_dropdown, direction_dropdown, seed=42, randomize_seed=False, guidance_scale=2.5):
    if randomize_seed:
        seed = random.randint(0, 2**32 - 1)
        
    input_image = input_image.convert("RGB")

    if illumination_dropdown != "custom" and prompt == "":
        prompt = ILLUMINATION_OPTIONS[illumination_dropdown]

    if illumination_dropdown != "custom" and prompt == ILLUMINATION_OPTIONS[illumination_dropdown]:
        prompt_prefix = f", with {illumination_dropdown}"
    else:
        prompt_prefix = ""

    if direction_dropdown != "auto" and prompt_prefix != "":
        prompt_prefix = prompt_prefix + f"coming from the {direction_dropdown} of the image"
    elif direction_dropdown != "auto" and prompt_prefix == "":
        prompt_prefix = f", light coming from the {direction_dropdown} of the image"
    
    prompt_with_template = f"Relight the image{prompt_prefix}. {prompt} Maintain the identity of the foreground subjects."
    
    pipe = Predict().setup()
    
    image = pipe.pipe(
        image=input_image, 
        prompt=prompt_with_template,
        guidance_scale=guidance_scale,
        width=input_image.size[0],
        height=input_image.size[1],
        generator=torch.Generator().manual_seed(seed),
    ).images[0]
    
    return image
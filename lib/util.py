import os
import torch
from PIL import ImageDraw, ImageFont, ImageStat
from torchvision import transforms
from .inceptionv3.meme_generation_lstm import MemeGenerationLSTM as MemeGenerationLSTMInceptionV3
from .resnet50.meme_generation_lstm import MemeGenerationLSTM as MemeGenerationLSTMResNet50
from .embedding import embedding, SPECIAL_TOKENS

# Function to overlay the generated meme on the images
cwd = os.getcwd()
FONT_PATH = cwd + "/fonts/impact.ttf"


def draw_caption(draw, text, position, font, image_width, font_color):
    text_length = draw.textlength(text, font=font)

    x = (image_width - text_length) // 2
    draw.text((x, position), text, font=font, fill=font_color)


def determine_font_size(txt, image):
    fontsize = 1  # starting font size

    # portion of image width you want text width to be
    img_fraction = 0.90

    font = ImageFont.truetype(FONT_PATH, fontsize)
    while font.getlength(txt) < img_fraction * image.size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype(FONT_PATH, fontsize)

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype(FONT_PATH, fontsize)
    return font


def overlay_captions(image, top_text, bottom_text):
    # Calculate the average brightness
    stat = ImageStat.Stat(image)
    avg_brightness = stat.mean[0]

    # Choose the font color based on the average brightness
    if avg_brightness > 130:
        font_color = "black"
    else:
        font_color = "white"

    # Draw the captions on the new image
    draw = ImageDraw.Draw(image)

    # Draw the top portion of the caption
    font = determine_font_size(top_text, image)
    draw_caption(draw, top_text, 10, font, image.width, font_color)

    if bottom_text:
        # Draw the bottom portion of the caption
        font = determine_font_size(bottom_text, image)
        _, top, _, bottom = font.getbbox(bottom_text)
        draw_caption(
            draw,
            bottom_text,
            image.height - bottom - top - 10,
            font,
            image.width,
            font_color,
        )

    return image


def get_model_instance(model_name):
    if model_name == "InceptionV3":
        model_path = f"{cwd}/models/MemeGenerationInceptionv3.best.pth"
        model_class = MemeGenerationLSTMInceptionV3
    else:
        model_path = f"{cwd}/models/MemeGenerationResnet.best.pth"
        model_class = MemeGenerationLSTMResNet50
    
    model = model_class.from_pretrained(model_path).eval()

    return model


def generate_meme(image, model_name):
    # The image pre-processing will be performed same as it was done during training
    img_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # this will convert the image to (C x H x W)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Open the Image and apply preprocessing what we applied during training
    processed_image = img_preprocess(image)

    # Load the model
    model_inference = get_model_instance(model_name)

    with torch.no_grad():
        output = model_inference.cpu().generate(
            processed_image.unsqueeze(0).cpu(),
            max_len=25,
            temperature=1.0,
            beam_width=10,
            top_k=10,
        )

        out = output[0]

        # Convert the token indexes to it's corresponding string and remove EOS
        generated_token = [
            embedding.itos[index.item()]
            for index in out
            if index.item() != 0
            and index.item() != embedding.stoi[SPECIAL_TOKENS["EOS"]]
        ]

        # Check if the token contains profane tokens hide last two characters
        profane_tokens = [
            "fuck",
            "fagot",
            "motherfucker",
            "fucking",
            "lesbian",
            "slut",
            "rape",
        ]

        for i, token in enumerate(generated_token):
            if token in profane_tokens:
                generated_token[i] = token[:-2] + "**"

        token_length = len(generated_token)

        if (
            token_length < 6
        ):  # If there are only 6 tokens, then it can be accomodated at the top portion of image
            top_meme_portion = " ".join(generated_token)
            bottom_meme_portion = ""
        else:
            top_meme_portion = " ".join(generated_token[: token_length // 2])
            bottom_meme_portion = " ".join(generated_token[token_length // 2 :])

        return overlay_captions(image, top_meme_portion, bottom_meme_portion)

import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET
from options import *

# Randomly select modules
# Get consistent skin tone across body parts
skin_tone = random.choice(['tint1', 'tint2', 'tint3', 'tint4', 'tint5', 'tint6', 'tint7', 'tint8'])

shirt_color = random.choice(['blue', 'green', 'grey', 'navy', 'pine', 'red', 'white', 'Yellow'])
shirt_name = random.choice([s for s in shirt_options if shirt_color.lower() in s.lower()]) + '.png'
sleeve_name = random.choice([s for s in shirt_sleeve_options if shirt_color.lower() in s.lower()]) + '.png'

pants_color = random.choice(['pantsBlue1', 'pantsBlue2', 'Brown', 'Green', 'Grey', 'LightBlue', 'Navy', 'Pine', 'Red', 'Tan', 'White', 'Yellow'])
pants_name = random.choice([p for p in pants_leg_options if pants_color in p]) + '.png'
pants_top_name = random.choice([p for p in pants_top_options if pants_color in p]) + '.png'

shoe_name = random.choice(list(shoe_options)) + '.png'
hair_name = random.choice(list(hair_options)) + '.png'
face_name = random.choice(list(face_options)) + '.png'


arm_name = f"{skin_tone}_arm.png"
hand_name = f"{skin_tone}_hand.png"
head_name = f"{skin_tone}_head.png"
leg_name = f"{skin_tone}_leg.png"
neck_name = f"{skin_tone}_neck.png"



# Load an image with alpha channel if available; add alpha if missing.
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = 255 * np.ones(b.shape, dtype=b.dtype)
        img = cv2.merge([b, g, r, alpha])
    return img

# Overlay function that blends fg onto bg with transparency.
def overlay_image(bg, fg, x, y):
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]
    
    if x < 0:
        fg = fg[:, -x:]
        fw = fg.shape[1]
        x = 0
    if y < 0:
        fg = fg[-y:, :]
        fh = fg.shape[0]
        y = 0
    if x + fw > bw:
        fw = bw - x
        fg = fg[:, :fw]
    if y + fh > bh:
        fh = bh - y
        fg = fg[:fh, :]
    
    roi = bg[y:y+fh, x:x+fw]
    
    fg = fg.astype(float)
    roi = roi.astype(float)
    alpha_fg = fg[:, :, 3:4] / 255.0
    alpha_bg = roi[:, :, 3:4] / 255.0
    
    blended = fg[:, :, :3] * alpha_fg + roi[:, :, :3] * (1 - alpha_fg)
    new_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
    
    roi[:, :, :3] = blended
    roi[:, :, 3:4] = new_alpha * 255.0
    bg[y:y+fh, x:x+fw] = roi.astype(np.uint8)
    return bg

# Helper: compute top-left from center (cx, cy) and overlay fg.
def overlay_image_center(bg, fg, cx, cy):
    fh, fw = fg.shape[:2]
    x = int(cx - fw / 2)
    y = int(cy - fh / 2)
    return overlay_image(bg, fg, x, y)

# Load a texture atlas given an XML and image path.
# Returns a dict mapping subtexture names to the cropped image.
def load_atlas(xml_path, image_path):
    atlas = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    sheet = load_img(image_path)
    for sub in root.findall('SubTexture'):
        name = sub.attrib['name']
        x = int(sub.attrib['x'])
        y = int(sub.attrib['y'])
        width = int(sub.attrib['width'])
        height = int(sub.attrib['height'])
        # Crop the region; note that sheet is in (y,x) order.
        sub_img = sheet[y:y+height, x:x+width].copy()
        atlas[name] = sub_img
    return atlas

base_path = 'kenney_modular-characters/Spritesheet/'
# Load all atlases.
modules_skin   = load_atlas(f"{base_path}sheet_skin.xml",   f"{base_path}sheet_skin.png")
modules_shoes  = load_atlas(f"{base_path}sheet_shoes.xml",  f"{base_path}sheet_shoes.png")
modules_shirts = load_atlas(f"{base_path}sheet_shirts.xml", f"{base_path}sheet_shirts.png")
modules_pants  = load_atlas(f"{base_path}sheet_pants.xml",  f"{base_path}sheet_pants.png")
modules_hair   = load_atlas(f"{base_path}sheet_hair.xml",   f"{base_path}sheet_hair.png")
modules_face   = load_atlas(f"{base_path}sheet_face.xml",   f"{base_path}sheet_face.png")


# Now extract the required modules.
head            = modules_skin[head_name]
arm_right       = modules_skin[arm_name]
hand_right      = modules_skin[hand_name]
leg_right       = modules_skin[leg_name]
neck            = modules_skin[neck_name]
shirt           = modules_shirts[shirt_name]
sleeve_right    = modules_shirts[sleeve_name]
pants_right     = modules_pants[pants_name]
pants_top       = modules_pants[pants_top_name]
shoe_right      = modules_shoes[shoe_name]
hair            = modules_hair[hair_name]
face            = modules_face[face_name]

# Create flipped versions for left side parts.
pants_left  = cv2.flip(pants_right, 1)
hand_left   = cv2.flip(hand_right, 1)
sleeve_left = cv2.flip(sleeve_right, 1)
shoe_left   = cv2.flip(shoe_right, 1)
leg_left    = cv2.flip(leg_right, 1)
arm_left    = cv2.flip(arm_right, 1)

# Create a blank canvas (RGBA, transparent).
canvas = np.zeros((600, 600, 4), dtype=np.uint8)

# Define center positions for each part.
offsets = {
    "hair": (0, 0), # updated later based on hair name
    "pants": (353, 470) if 'long' in pants_name else (350, 441) if 'shorter' in pants_name else (351, 445),
    "shirt": (300, 290),
    "arm": (433, 280),
    "neck": (300, 215),
    "sleeve": (430, 275) if 'long' in sleeve_name else (378, 247) if 'shorter' in sleeve_name else (403, 252),
    "leg": (350, 470),
    "hand": (510, 350),
    "head": (300, 125),
    "face": (300, 140),
    "shoe": (380, 560),
    "pants_top": (300, 394),
}

# Update hair offset based on name
if 'Man' in hair_name and '5' not in hair_name:
    offsets["hair"] = (300, 80)
elif 'Man5' in hair_name:
    offsets["hair"] = (310, 80)
elif 'Woman1' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (300, 140)
    else:
        offsets["hair"] = (300, 168)
elif 'Woman2' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (300, 168)
    else:
        offsets["hair"] = (300, 144)
elif 'Woman3' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (300, 144)
    elif 'tan' in hair_name:
        offsets["hair"] = (301, 133)
    else:
        offsets["hair"] = (302, 170)
elif 'Woman4' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (302, 170)
    elif 'tan' in hair_name:
        offsets["hair"] = (302, 170)
    else:
        offsets["hair"] = (300, 140)
elif 'Woman5' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (301, 133)
    elif 'tan' in hair_name:
        offsets["hair"] = (300, 140)
    else:
        offsets["hair"] = (300, 136)
elif 'Woman6' in hair_name:
    if 'black' in hair_name:
        offsets["hair"] = (300, 136)
    elif 'tan' in hair_name:
        offsets["hair"] = (300, 136)
    else:
        offsets["hair"] = (301, 133)

offsets.update({
    "pants_left": (600-offsets["pants"][0], offsets["pants"][1]),
    "hand_left": (600-offsets["hand"][0], offsets["hand"][1]),
    "sleeve_left": (600-offsets["sleeve"][0], offsets["sleeve"][1]),
    "shoe_left": (600-offsets["shoe"][0], offsets["shoe"][1]),
    "leg_left": (600-offsets["leg"][0], offsets["leg"][1]),
    "arm_left": (600-offsets["arm"][0], offsets["arm"][1]),
})

# Overlay each part using center coordinates.
canvas = overlay_image_center(canvas, neck, *offsets["neck"])
canvas = overlay_image_center(canvas, arm_right, *offsets["arm"])
canvas = overlay_image_center(canvas, arm_left, *offsets["arm_left"])
canvas = overlay_image_center(canvas, hand_right, *offsets["hand"])
canvas = overlay_image_center(canvas, hand_left, *offsets["hand_left"])
canvas = overlay_image_center(canvas, sleeve_right, *offsets["sleeve"])
canvas = overlay_image_center(canvas, sleeve_left, *offsets["sleeve_left"])
canvas = overlay_image_center(canvas, shirt, *offsets["shirt"])
canvas = overlay_image_center(canvas, leg_right, *offsets["leg"])
canvas = overlay_image_center(canvas, leg_left, *offsets["leg_left"])
canvas = overlay_image_center(canvas, pants_right, *offsets["pants"])
canvas = overlay_image_center(canvas, pants_left, *offsets["pants_left"])
canvas = overlay_image_center(canvas, pants_top, *offsets["pants_top"])
canvas = overlay_image_center(canvas, head, *offsets["head"])
canvas = overlay_image_center(canvas, face, *offsets["face"])
canvas = overlay_image_center(canvas, hair, *offsets["hair"])
canvas = overlay_image_center(canvas, shoe_right, *offsets["shoe"])
canvas = overlay_image_center(canvas, shoe_left, *offsets["shoe_left"])

# Show and save the final composite image.
cv2.destroyAllWindows()
cv2.imwrite("charac.png", canvas)

print('Created character:')
print(f"skin_tone = '{skin_tone}'")
print(f"shirt_name = '{shirt_name}'")
print(f"sleeve_name = '{sleeve_name}'")
print(f"pants_name = '{pants_name}'")
print(f"pants_top_name = '{pants_top_name}'")
print(f"shoe_name = '{shoe_name}'")
print(f"hair_name = '{hair_name}'")
print(f"face_name = '{face_name}'")
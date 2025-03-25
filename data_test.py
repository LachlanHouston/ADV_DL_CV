import cv2
import numpy as np

# Load image with alpha channel if available; add alpha if missing.
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Image not found: {path}")
    # If image has 3 channels (BGR), add an alpha channel (full opacity)
    if len(img.shape) == 3 and img.shape[2] == 3:
        b, g, r = cv2.split(img)
        alpha = 255 * np.ones(b.shape, dtype=b.dtype)
        img = cv2.merge([b, g, r, alpha])
    return img

# Overlay function that blends fg onto bg with transparency, handling clipping.
def overlay_image(bg, fg, x, y):
    fh, fw = fg.shape[:2]
    bh, bw = bg.shape[:2]
    
    # Clip the overlay if it goes out of the background bounds.
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
    
    # Convert to float for blending.
    fg = fg.astype(float)
    roi = roi.astype(float)
    
    # Normalize alpha channels.
    alpha_fg = fg[:, :, 3:4] / 255.0
    alpha_bg = roi[:, :, 3:4] / 255.0
    
    # Blend the color channels.
    blended = fg[:, :, :3] * alpha_fg + roi[:, :, :3] * (1 - alpha_fg)
    # Compute new alpha.
    new_alpha = alpha_fg + alpha_bg * (1 - alpha_fg)
    
    # Write blended colors and new alpha back to ROI.
    roi[:, :, :3] = blended
    roi[:, :, 3:4] = new_alpha * 255.0
    bg[y:y+fh, x:x+fw] = roi.astype(np.uint8)
    return bg

# New helper: Compute top-left from center (cx, cy) and overlay fg.
def overlay_image_center(bg, fg, cx, cy):
    fh, fw = fg.shape[:2]
    x = int(cx - fw / 2)
    y = int(cy - fh / 2)
    return overlay_image(bg, fg, x, y)

# Paths to your images (note: "hair" is the actual hair, not a base)
paths = {
    "head": "1example/tint1_head.png",
    "face": "1example/face1.png",
    "hair": "1example/blackMan1.png",
    "arm": "1example/tint1_arm.png",
    "shirt": "1example/blueShirt1.png",
    "sleeve": "1example/blueArm_long.png",
    "neck": "1example/tint1_neck.png",
    "leg": "1example/tint1_leg.png",
    "hand": "1example/tint1_hand.png",
    "pants": "1example/pantsBlue1_long.png",
    "shoe": "1example/blackShoe1.png",
    "pants_top": "1example/pantsBlue11.png",
}

# Load all images.
hair = load_img(paths["hair"])
pants = load_img(paths["pants"])
pants_top = load_img(paths["pants_top"])
shirt = load_img(paths["shirt"])
arm = load_img(paths["arm"])
neck = load_img(paths["neck"])
sleeve = load_img(paths["sleeve"])
leg = load_img(paths["leg"])
head = load_img(paths["head"])
hand = load_img(paths["hand"])
face = load_img(paths["face"])
shoe = load_img(paths["shoe"])

# Flip the 'pants' image horizontally to create the left leg.
pants_left = cv2.flip(pants, 1)  # 1 is the flag to flip horizontally.
hand_left = cv2.flip(hand, 1)  # Flip the hand image as well.
sleeve_left = cv2.flip(sleeve, 1)  # Flip the sleeve image as well.
shoe_left = cv2.flip(shoe, 1)  # Flip the shoe image as well.
leg_left = cv2.flip(leg, 1)  # Flip the leg image as well.
arm_left = cv2.flip(arm, 1)  # Flip the arm image as well.

# Create a blank 300x300 canvas (RGBA, transparent).
h, w = 600, 600
canvas = np.zeros((h, w, 4), dtype=np.uint8)

# Define center positions for each part.
# Adjust these as needed for correct alignment.
offsets = {
    "hair": (300, 80),
    "pants": (353, 470),
    "shirt": (300, 290),
    "arm": (433, 280),
    "neck": (300, 215),
    "sleeve": (430, 275),
    "leg": (350, 470),
    "hand": (510, 350),
    "head": (300, 125), 
    "face": (300, 140),
    "shoe": (380, 560),
    "pants_top": (300, 394),
}
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
canvas = overlay_image_center(canvas, shirt, *offsets["shirt"])
canvas = overlay_image_center(canvas, head, *offsets["head"])
canvas = overlay_image_center(canvas, face, *offsets["face"])
canvas = overlay_image_center(canvas, hair, *offsets["hair"])
canvas = overlay_image_center(canvas, arm, *offsets["arm"])
canvas = overlay_image_center(canvas, arm_left, *offsets["arm_left"])
canvas = overlay_image_center(canvas, hand, *offsets["hand"])
canvas = overlay_image_center(canvas, hand_left, *offsets["hand_left"])
canvas = overlay_image_center(canvas, sleeve, *offsets["sleeve"])
canvas = overlay_image_center(canvas, sleeve_left, *offsets["sleeve_left"])
canvas = overlay_image_center(canvas, leg, *offsets["leg"])
canvas = overlay_image_center(canvas, leg_left, *offsets["leg_left"])
canvas = overlay_image_center(canvas, pants_top, *offsets["pants_top"])
canvas = overlay_image_center(canvas, pants, *offsets["pants"])
canvas = overlay_image_center(canvas, pants_left, *offsets["pants_left"])
canvas = overlay_image_center(canvas, shoe, *offsets["shoe"])
canvas = overlay_image_center(canvas, shoe_left, *offsets["shoe_left"])

# Display the final composite image.
cv2.imshow('Complete Character', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output.
cv2.imwrite("complete_character.png", canvas)
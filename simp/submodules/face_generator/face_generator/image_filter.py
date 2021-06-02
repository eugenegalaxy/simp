from PIL import Image, ImageEnhance, ImageFilter


img_path = "sample/hugo_cropped.jpg"
img = Image.open(img_path)

# enhancer = ImageEnhance.Contrast(img)
# factor = 1.2  # increase contrast
# img = enhancer.enhance(factor)


left = 260
top = 0
right = 520
bottom = 400
  

img = img.crop((left, top, right, bottom))
img.save(img_path)
img.show()

# img_filtered = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
# img_filtered.save(img_path)
# img_filtered.show()

from PIL import Image

def resize_image(image_path, new_path, new_size):
    with Image.open(image_path) as img:
        imgResize = img.resize(new_size)
        imgResize.save(new_path)

if __name__ == "__main__":
    resize_image('/Users/leohsuinthehouse/Desktop/sketch.jpg', '/Users/leohsuinthehouse/Desktop/resize_sketch.jpg', (512, 512))
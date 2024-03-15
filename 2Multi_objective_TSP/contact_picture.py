import os
from PIL import Image, ImageDraw, ImageFont
import re

def parse_filename(filename):
    pattern = r'pop_size:(\d+)_num_generations:(\d+)_mutation_rate:(\d+\.\d+)_crossover_rate:(\d+\.\d+)\.png'
    match = re.match(pattern, filename)
    if match:
        pop_size = int(match.group(1))
        num_generations = int(match.group(2))
        mutation_rate = float(match.group(3))
        crossover_rate = float(match.group(4))
        return (pop_size, num_generations, mutation_rate, crossover_rate)
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

# 获取当前文件夹中所有的png文件
files = [f for f in os.listdir('.') if f.endswith('.png')]

# 将文件按照参数顺序排序
files.sort(key=parse_filename)

# 每行文件数量
files_per_row = 6

# 加载一个字体
try:
    font = ImageFont.truetype("arial.ttf", size=30)
except IOError:
    font = ImageFont.load_default()

# 创建一个新的图片对象，用于拼接，并且考虑到文字的高度
images_with_text = []
text_height = 40  # 留出足够的空间来写文字
for f in files:
    image = Image.open(f)
    # 创建一个新的图像，高度增加了text_height
    new_height = image.size[1] + text_height
    new_image = Image.new("RGB", (image.size[0], new_height), (255, 255, 255))
    new_image.paste(image, (0, 0))
    draw = ImageDraw.Draw(new_image)
    params_str = f.replace('.png', '').replace('_', ', ')
    # 将参数文字写在新图像下方的白色区域
    draw.rectangle([(0, image.size[1]), (image.size[0], new_height)], fill="white")
    draw.text((5, image.size[1] + 5), params_str, font=font, fill="black")
    images_with_text.append(new_image)

widths, heights = zip(*(i.size for i in images_with_text))

# 计算最大宽度和总高度
max_width = max(widths)
total_height = sum([heights[i] for i in range(0, len(heights), files_per_row)])

# 创建一个足够大的空白图片来容纳所有拼接后的图片和文字
new_im = Image.new('RGB', (max_width * files_per_row, total_height), (255, 255, 255))

y_offset = 0
for i in range(0, len(images_with_text), files_per_row):
    x_offset = 0
    row_height = max(heights[i:i + files_per_row])
    for im in images_with_text[i:i + files_per_row]:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += max_width
    y_offset += row_height

# 保存拼接后的图片
new_im.save('combined_image.png')
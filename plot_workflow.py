import glob

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display

def concatenate_images_with_text(images_left, images_right, label):
    # Assume all images have the same size (width, height)
    image_width, image_height = images_left[0].size

    # Additional space for the text "--generated->" between the images
    space_between_images = 700  # Adjust this value to fit the text comfortably

    # Create a blank image for the row plus text label
    total_width = (len(images_left) + len(images_right)) * image_width + space_between_images
    total_height = image_height * 2  # Double the height to fit images and text
    new_row = Image.new('RGB', (total_width, total_height), (255, 255, 255))  # White background

    draw = ImageDraw.Draw(new_row)

    # Load a font for the text (you might need to provide the font file path)
    try:
        font = ImageFont.truetype("arial.ttf", 100)  # You can adjust the font size
    except IOError:
        font = ImageFont.load_default()

    # Paste the left images (4 images)
    x_offset = 0
    for img in images_left:
        new_row.paste(img, (x_offset, 0))
        x_offset += image_width

    # Add the text "--generated->" in the middle
    generated_text = "--generate->"
    text_bbox = draw.textbbox((0, 0), generated_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Center the text vertically with the images and horizontally in the space between left and right images
    text_position = (x_offset + (space_between_images - text_width) // 2, (image_height - text_height) // 2)
    draw.text(text_position, generated_text, fill="black", font=font)

    # Move the x_offset to account for the text space
    x_offset += space_between_images

    # Paste the right images (4 images)
    for img in images_right:
        new_row.paste(img, (x_offset, 0))
        x_offset += image_width

    # Get the bounding box for the row's label text (below the images)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    label_text_width, label_text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

    # Draw the label text below the images
    label_text_position = (
    image_width * 5.5 , image_height + label_text_height)
    draw.text(label_text_position, label, fill="black", font=font)

    return new_row

def concatenate_rows_with_texts(all_rows):
    row_width, row_height = all_rows[0].size
    total_width = row_width
    total_height = len(all_rows) * row_height

    # Create a blank image for the final concatenated result
    final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Paste each row one below the other
    y_offset = 0
    for row in all_rows:
        final_image.paste(row, (0, y_offset))
        y_offset += row_height

    return final_image

if __name__ == '__main__':

    # Load your images (replace with actual paths to your image files)
    dataset = 'myfriends'
    name = 'chengyu'
    attack_name = 'glaze'
    defense_names = ['bf', 'gn', 'bf_gn', 'gn_bf', 'pdmpure', 'diffpure', 'adavoc', 'ape']
    left_images_row0 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}/set_B/*')]
    img_size = left_images_row0[0].size
    right_images_row0 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row1 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}/*')]
    right_images_row1 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row2 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[0]}/*')]
    right_images_row2 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[0]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row3 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[1]}/*')]
    right_images_row3 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[1]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row4 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[2]}/*')]
    right_images_row4 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[2]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row5 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[3]}/*')]
    right_images_row5 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[3]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row6 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[4]}/*')]

    right_images_row6 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[4]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row7 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[5]}/*')]
    right_images_row7 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[5]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    for i in range(len(left_images_row7)):
        left_images_row7[i] = left_images_row7[i].resize(img_size)
    for i in range(len(right_images_row7)):
        right_images_row7[i] = right_images_row7[i].resize(img_size)

    left_images_row8 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[6]}/*')]
    right_images_row8 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[6]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    left_images_row9 = [Image.open(i) for i in glob.glob(f'dataset/{dataset}/{name}_{attack_name}_{defense_names[7]}/*')]
    right_images_row9 = [Image.open(i) for i in glob.glob(f'dreambooth-outputs/{name}_{attack_name}_{defense_names[7]}/checkpoint-1000/dreambooth/a_photo_of_sks_person/*')[:16]]

    df = pd.read_csv("result/result.csv")
    # Concatenate images with labels
    row0 = concatenate_images_with_text(left_images_row0, right_images_row0, "original  generated ISM: %.2f; FDR: %.2f" % (df.loc[df['img'] == name, 'ism'].values[0], df.loc[df['img'] == name, 'fdr'].values[0]))
    row1 = concatenate_images_with_text(left_images_row1, right_images_row1, "%s generated ISM: %.2f; FDR: %.2f" % (attack_name, df.loc[df['img'] == f"{name}_{attack_name}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}", 'fdr'].values[0]))
    row2 = concatenate_images_with_text(left_images_row2, right_images_row2, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[0]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[0]}", 'fdr'].values[0]))
    row3 = concatenate_images_with_text(left_images_row3, right_images_row3, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[1], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[1]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[1]}", 'fdr'].values[0]))
    row4 = concatenate_images_with_text(left_images_row4, right_images_row4, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[2], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[2]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[2]}", 'fdr'].values[0]))
    row5 = concatenate_images_with_text(left_images_row5, right_images_row5, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[3], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[3]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[3]}", 'fdr'].values[0]))
    row6 = concatenate_images_with_text(left_images_row6, right_images_row6, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[4], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[4]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[4]}", 'fdr'].values[0]))
    row7 = concatenate_images_with_text(left_images_row7, right_images_row7, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[5], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[5]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[5]}", 'fdr'].values[0]))
    row8 = concatenate_images_with_text(left_images_row8, right_images_row8, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[6], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[6]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[6]}", 'fdr'].values[0]))
    row9 = concatenate_images_with_text(left_images_row9, right_images_row9, "%s generated ISM: %.2f; FDR: %.2f" % (defense_names[7], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[7]}", 'ism'].values[0], df.loc[df['img'] == f"{name}_{attack_name}_{defense_names[7]}", 'fdr'].values[0]))
    # Concatenate all rows into the final image
    final_image = concatenate_rows_with_texts([row0, row1, row2, row3, row4, row5, row6, row7, row8, row9])

    # Save or display the final image
    final_image.save(f'graphs/concatenated_image_with_text_{name}_{attack_name}.jpg')
    # display(final_image)
"""Compare the saved environment render with its real-photo crop.

Run after render_dataset.py, or reuse existing results. Camera poses and piece
placements live in config.json; this script never changes them.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
# Original-photo crop and grid corners, ordered a1, h1, h8, a8.
CROP = [605, 580, 1090, 940]
CORNERS = [[662, 872], [1026, 849], [961, 612], [658, 625]]


def compare():
    """Assemble the actual photo crop and render without warping either image."""
    from PIL import Image, ImageDraw, ImageFont
    config = json.loads((HERE / 'config.json').read_text())
    output = ROOT / config['output']
    render = Image.open(output / 'images/test/environment_photo.png').convert('RGB')
    real = Image.open(PACKAGE / 'references/environment.jpg').convert('RGB')
    real = real.crop(CROP).resize(render.size, Image.Resampling.LANCZOS)
    real.save(output / 'reference_crop.png')
    width, height = render.size
    sheet = Image.new('RGB', (width*2, height+108), (25,29,34))
    draw = ImageDraw.Draw(sheet)
    font_path = Path('/usr/share/fonts/vazirmatn-vf-fonts/Vazirmatn[wght].ttf')
    heading = ImageFont.truetype(str(font_path), 28) if font_path.exists() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    draw.text((20,10), 'Real photo — environment.jpg', font=heading, fill='white')
    draw.text((width+20,10), 'Blender — matched camera and 18-piece arrangement', font=heading, fill='white')
    sheet.paste(real, (0,54)); sheet.paste(render, (width,54))
    draw.text((20,height+68), 'Lighting, piece identification and rotations are estimates. No image warping or retouching.', font=small, fill='white')
    sheet.save(output/'comparison.jpg', quality=95)
    Image.blend(real,render,.5).save(output/'alignment_overlay.png')
    record = json.loads((output/'records/test/environment_photo.json').read_text())
    # The renderer already records projected grid corners in image pixels.
    assert record['corners_order'] == ['a1 outer', 'h1 outer', 'h8 outer', 'a8 outer']
    pixels = np.array(record['corners'])
    expected = (np.array(CORNERS)-CROP[:2])*2
    rms = float(np.sqrt(np.mean((pixels-expected)**2)))
    assert rms < 2, pixels
    assert len(record['instances']) == 18
    notes = {'original_crop_pixels':CROP, 'display_upscale':2,
             'render_corner_rms_pixels':rms,
             'piece_count':18, 'uncertain_identification':['g7 bishop'],
             'orientation_note':'Knight headings and king cross rotations estimated; bishop/rook yaw uncertain.',
             'limitations':['Room, robots and off-board objects are not reconstructed.',
                            'Planar camera fit assumes centered principal point in original photo and no distortion.',
                            'Lighting/material parameters are not uniquely recoverable from one image.']}
    (output/'comparison_notes.json').write_text(json.dumps(notes,indent=2)+'\n')
    print('Comparison saved:',output/'comparison.jpg')
    print('Actual Blender corner RMS:',rms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', action='store_true', help='Optional alias for the default comparison action')
    parser.parse_args()
    compare()

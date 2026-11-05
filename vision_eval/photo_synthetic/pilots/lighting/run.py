"""Render controlled lighting comparisons using the existing paired-view renderer.

Run with system Python. Every variant keeps the same FEN, cameras and random seed.
Outputs are separate datasets with their exact config and an editable scene.blend.
"""
import argparse
import copy
import json
import html
from pathlib import Path
import subprocess
import sys
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
from check_lighting import check as check_lighting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--base-config', type=Path, help='Use these positions/cameras with the lighting presets')
    parser.add_argument('--output', default='vision_eval/photo_synthetic/pilots/lighting/results')
    parser.add_argument('--threads', type=int, default=4, help='CPU threads per Blender render')
    args = parser.parse_args()
    tests = json.loads(args.config.read_text())
    if 'presets_file' in tests:
        presets = json.loads((args.config.parent / tests['presets_file']).read_text())
        tests = {**presets, **tests}
    reference_base = json.loads((args.config.parent / tests['base_config']).read_text())
    base = json.loads(args.base_config.read_text()) if args.base_config else reference_base
    reference = tests.get('reference_comparison') if args.base_config is None else None
    if reference and not (PACKAGE / reference['image']).is_file():
        raise FileNotFoundError(f"Missing comparison photo: {PACKAGE / reference['image']}")
    if args.base_config:
        # The validated lighting depends on the calibrated finishes and camera response.
        for key in ['surface_finishes', 'wood', 'view_transform', 'view_look']:
            if key in reference_base:
                base[key] = copy.deepcopy(reference_base[key])
    output = (ROOT / args.output).resolve()
    assert output.is_relative_to(ROOT), 'Output must stay inside the project'
    # Refuse overwrites so earlier comparisons remain available.
    output.mkdir(parents=True, exist_ok=False)
    snapshot = dict(tests, base_config='base_config.json')
    snapshot.pop('presets_file', None)  # Snapshot embeds presets and can be replayed alone.
    (output / 'settings.json').write_text(json.dumps(snapshot, indent=2))
    (output / 'base_config.json').write_text(json.dumps(base, indent=2))
    cards = []
    previews = {view: [] for view in base['cameras']}
    for variant in tests['variants']:
        name = variant['id']
        assert name and Path(name).name == name and name not in {'.', '..'}
        config = copy.deepcopy(base)
        config.update(variant['overrides'])
        # This runner compares exact presets, even if a supplied batch randomizes them.
        config.pop('lighting_randomization', None)
        config['lighting_note'] = tests.get('note', '') + ' Variant: ' + name + '. ' + variant.get('description', '')
        config['output'] = str((output / name).relative_to(ROOT))
        config['samples'] = tests['samples']
        # Disable all placement/camera jitter: only the tested settings may change.
        for key in ['piece_jitter_mm', 'piece_yaw_jitter_degrees',
                    'camera_position_jitter_mm', 'camera_target_jitter_mm']:
            config[key] = 0
        for camera in config['cameras'].values():
            camera['resolution'] = [
                max(1, round(n * tests['resolution_scale']))
                for n in camera.get('resolution', config['resolution'])
            ]
        path = output / (name + '.json')
        path.write_text(json.dumps(config, indent=2))
        print('Rendering', name, flush=True)
        with (output / (name + '.log')).open('w') as log:
            subprocess.run([
                str(ROOT / '.local-blender/start-blender.sh'), '--background', '--threads', str(args.threads), '--python-exit-code', '1',
                '--python', str(PACKAGE / 'render_dataset.py'), '--', '--config', str(path),
            ], stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        # Validate each exported pair before including it in the comparison.
        subprocess.run([sys.executable, str(PACKAGE / 'check_dataset.py'), str(output / name)],
                       check=True, cwd=ROOT)
        if 'quality_limits' in tests:
            check_lighting(output / name, tests['quality_limits'])
        for view in config['cameras']:
            image = Path(name) / 'images' / config['positions'][0]['split'] / (
                config['positions'][0]['id'] + '_' + view + '.png'
            )
            assert (output / image).is_file(), image
            previews[view].append((name, output / image))
            cards.append(f'<figure><figcaption>{html.escape(name)} — {html.escape(view)}</figcaption>'
                         f'<img src="{html.escape(image.as_posix(), quote=True)}"></figure>')
        # The variant folder retains its exact config; keep logs only on failure.
        path.unlink()
        (output / (name + '.log')).unlink()
    (output / 'comparison.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Lighting comparisons</title>'
        '<style>body{background:#222;color:white;font:16px sans-serif}'
        'main{display:grid;grid-template-columns:1fr 1fr;gap:20px}'
        'figure{margin:0}img{width:100%;height:440px;object-fit:contain}</style>'
        '<h1>Lighting comparisons — fixed and arm cameras</h1><main>' + ''.join(cards) + '</main>'
    )
    # Static contact sheets are convenient to review without opening the HTML.
    font_path = Path('/usr/share/fonts/vazirmatn-vf-fonts/Vazirmatn[wght].ttf')
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    for view, entries in previews.items():
        resolution = base['cameras'][view].get('resolution', base['resolution'])
        row_height = min(410, round(460 * resolution[1] / resolution[0])) + 50
        sheet = Image.new('RGB', (1440, row_height * ((len(entries) + 2) // 3)), '#19232d')
        draw = ImageDraw.Draw(sheet)
        for i, (name, path) in enumerate(entries):
            with Image.open(path) as image:
                image = image.convert('RGB')
                image.thumbnail((460, 410))
                x, y = (i % 3) * 480, (i // 3) * row_height
                draw.text((x + 10, y + 8), name.replace('_', ' '), font=font, fill='white')
                sheet.paste(image, (x + (480 - image.width) // 2, y + 40))
        sheet.save(output / f'comparison_{view}.jpg', quality=95)
    if reference:
        # Compare the original photo with an actual render; no warp or colour edits.
        variant = output / reference['variant']
        position = base['positions'][0]
        rendered = Image.open(variant / 'images' / position['split'] /
                              (position['id'] + '_' + reference['view'] + '.png')).convert('RGB')
        photo = Image.open(PACKAGE / reference['image']).convert('RGB')
        photo = photo.resize(rendered.size, Image.Resampling.LANCZOS)
        width, height = rendered.size
        sheet = Image.new('RGB', (width * 2, height + 90), '#19232d')
        draw = ImageDraw.Draw(sheet)
        draw.text((15, 8), 'Real Camera 2 reference (original overlays)', font=font, fill='white')
        draw.text((width + 15, 8), 'Blender — matched fixed-camera reference', font=font, fill='white')
        sheet.paste(photo, (0, 40)); sheet.paste(rendered, (width, 40))
        draw.text((15, height + 52), 'Partial visible-piece reconstruction; hidden pieces and poses are uncertain.', font=font, fill='white')
        sheet.save(output / 'reference_comparison.jpg', quality=95)
    print('Comparisons saved in', output)


if __name__ == '__main__':
    main()

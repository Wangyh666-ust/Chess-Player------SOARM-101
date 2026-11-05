"""Reject washed-out black squares or poorly lit boards in rendered datasets.

Samples the interior of unoccupied squares using the saved camera matrices.
Thresholds describe displayed PNG brightness, not measured lux or linear radiance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def projected_region(points, matrix, mask, require_full_view=False):
    """Rasterize a plane patch; return only pixels not covered by a chess piece."""
    height, width = mask.shape
    clip = np.array(points) @ matrix.T
    if np.any(clip[:, 3] <= 0):
        return np.zeros_like(mask, dtype=bool)
    ndc = clip[:, :2] / clip[:, 3, None]
    pixels = np.column_stack(((ndc[:, 0] + 1) * width / 2,
                              (1 - ndc[:, 1]) * height / 2))
    if require_full_view and (np.any(pixels < 0) or np.any(pixels >= [width, height])):
        return np.zeros_like(mask, dtype=bool)
    region = Image.new('1', (width, height))
    ImageDraw.Draw(region).polygon([tuple(p) for p in pixels], fill=1)
    return np.asarray(region) & (mask == 0)


def check(root, limits):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['samples'], 'No images to validate'
    dimensions = manifest['board_dimensions_mm']
    pitch = dimensions['square_mm'] / 1000
    z = (dimensions['thickness_mm'] + 0.06) / 1000
    report = {'limits': limits, 'samples': [], 'passed': True}
    for sample in manifest['samples']:
        name, split = sample['id'], sample['split']
        record = json.loads((root / 'records' / split / (name + '.json')).read_text())
        rgb = np.asarray(Image.open(root / 'images' / split / (name + '.png')).convert('RGB'))
        mask = np.asarray(Image.open(root / 'masks' / split / (name + '.png')))
        # RGB antialiasing blends bright piece edges into background pixels even
        # though the unfiltered instance mask says zero. Exclude that 2 px halo.
        mask = np.asarray(Image.fromarray((mask > 0).astype('uint8') * 255)
                          .filter(ImageFilter.MaxFilter(5)))
        height, width = mask.shape
        brightness = rgb @ np.array([0.2126, 0.7152, 0.0722])
        matrix = np.array(record['projection_matrix']) @ np.linalg.inv(record['camera_to_world'])
        occupied = {i['square'] for i in record['instances']}
        black, white = [], []
        for rank in range(8):
            for file in range(8):
                if 'abcdefgh'[file] + str(rank + 1) in occupied:
                    continue
                # Inset avoids grid edges and antialiasing; masks exclude occluders.
                points = np.array([[(file + x - 4) * pitch, (rank + y - 4) * pitch, z, 1]
                                   for x, y in [(0.22, 0.22), (0.78, 0.22), (0.78, 0.78), (0.22, 0.78)]])
                visible = projected_region(points, matrix, mask, require_full_view=True)
                values = brightness[visible]
                if len(values) < 24:
                    continue
                if (file + rank) % 2 == 0:  # a1 is black.
                    black.append(float(np.percentile(values, 95)))
                else:
                    # Upper percentile tolerates printed ArUco patterns on white cells.
                    white.append(float(np.percentile(values, 90)))
        if min(len(black), len(white)) < 4:
            # Crowded/low-resolution views can be unscorable. Record the reason
            # and fail the batch rather than treating missing evidence as a pass.
            report['samples'].append(dict(sample=name, black_squares=len(black),
                                          white_squares=len(white), passed=False,
                                          reason='Too few visible empty squares for lighting QA'))
            report['passed'] = False
            continue
        metrics = dict(sample=name, black_squares=len(black), white_squares=len(white),
                       black_max_p95=max(black), white_median_p90=float(np.median(white)),
                       white_spread_p90=max(white) - min(white))
        metrics['passed'] = (metrics['black_max_p95'] <= limits['max_black_square_p95']
                             and limits['min_white_median'] <= metrics['white_median_p90'] <= limits['max_white_median']
                             and metrics['white_spread_p90'] <= limits['max_white_spread'])
        if 'max_frame_p95' in limits:
            # The old square-only test missed glare on the frame and table.
            # Frame strips assume the current 7.5 mm frame; adjust after geometry edits.
            half_x, half_y = dimensions['width_mm'] / 2000, dimensions['depth_mm'] / 2000
            frame, surround = [], []
            for axis, half, other in [(0, half_x, half_y), (1, half_y, half_x)]:
                for sign in [-1, 1]:
                    for name, centre, patch_z, results in [
                        ('frame', half - .005, z + .0005, frame),
                        ('surround', (half - .0075 + 4 * pitch) / 2, z, surround),
                    ]:
                        points = []
                        for offset, along in [(-.001, -other + .009), (.001, -other + .009),
                                              (.001, other - .009), (-.001, other - .009)]:
                            xy = [sign * centre + offset, along]
                            if axis == 1:
                                xy.reverse()
                            points.append([*xy, patch_z, 1])
                        values = brightness[projected_region(points, matrix, mask)]
                        if len(values) >= 24:
                            results.append(float(np.percentile(values, 95)))
            outer = [[-half_x, -half_y, z, 1], [half_x, -half_y, z, 1],
                     [half_x, half_y, z, 1], [-half_x, half_y, z, 1]]
            outside = ~projected_region(outer, matrix, np.zeros_like(mask)) & (mask == 0)
            table = brightness[outside]
            for key, value, threshold in [
                ('frame_max_p95', max(frame) if frame else None, limits['max_frame_p95']),
                ('surround_max_p95', max(surround) if surround else None, limits['max_surround_p95']),
                ('table_p99', float(np.percentile(table, 99)) if len(table) >= 24 else None, limits['max_table_p99']),
            ]:
                metrics[key] = value
                metrics['passed'] &= value is None or value <= threshold
        report['samples'].append(metrics)
        report['passed'] &= metrics['passed']
    (root / 'lighting_qa.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['passed'], f'Lighting rejected; inspect {root / "lighting_qa.json"}'
    return report


def self_test(root, limits):
    """Verify acceptance and rejection using temporary copies of a good render."""
    import shutil
    import tempfile

    check(root, limits)
    with tempfile.TemporaryDirectory(prefix='chess-lighting-qa-') as directory:
        fixture = Path(directory)
        shutil.copy2(root / 'manifest.json', fixture / 'manifest.json')
        for folder in ['records', 'masks', 'images']:
            shutil.copytree(root / folder, fixture / folder)
        originals = {p: np.asarray(Image.open(p).convert('RGB'))
                     for p in (fixture / 'images').rglob('*.png')}
        # Deliberately faulty images exercise each visual rejection criterion.
        faults = ['washed_blacks', 'too_dark', 'too_bright', 'uneven_lighting']
        if 'max_frame_p95' in limits:
            faults.append('border_glare')
        for fault in faults:
            for path, rgb in originals.items():
                if fault == 'washed_blacks':
                    changed = np.maximum(rgb, 128)
                elif fault == 'too_dark':
                    changed = (rgb * 0.3).astype('uint8')
                elif fault == 'too_bright':
                    changed = np.full_like(rgb, 255)
                elif fault == 'uneven_lighting':
                    changed = rgb.copy()
                    changed[:, :rgb.shape[1] // 2] = rgb[:, :rgb.shape[1] // 2] * 0.5
                else:
                    record = json.loads((fixture / 'records' / path.parent.name / (path.stem + '.json')).read_text())
                    outside = Image.new('1', (rgb.shape[1], rgb.shape[0]), 1)
                    ImageDraw.Draw(outside).polygon([tuple(p) for p in record['corners']], fill=0)
                    changed = rgb.copy()
                    changed[np.asarray(outside)] = 245
                Image.fromarray(changed).save(path)
            try:
                check(fixture, limits)
            except AssertionError:
                report = json.loads((fixture / 'lighting_qa.json').read_text())
                assert not report['passed'], fault
            else:
                raise AssertionError(f'Failed to reject {fault}')
    print(f'Lighting QA self-test passed: good render accepted; {len(faults)} bad renders rejected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--presets', type=Path, default=Path(__file__).parent / 'configs/lighting_presets.json')
    parser.add_argument('--self-test', action='store_true', help='Test rejection on temporary faulty image copies')
    args = parser.parse_args()
    limits = json.loads(args.presets.read_text())['quality_limits']
    if args.self_test:
        self_test(args.root, limits)
    else:
        print(json.dumps(check(args.root, limits), indent=2))

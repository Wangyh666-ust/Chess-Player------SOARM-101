import copy
import sys

import cv2
import numpy as np

import score_grid_bridge_v1 as score
import live_weighted_grid_bridge_v2 as weighted
import guided_grid_scale_adapter_v1 as guided

score.GridLocator = guided.GridLocator


def _hidden_squares(grid):
    if grid is None or not isinstance(grid, dict):
        return None
    cells = grid.get("cells")
    if not isinstance(cells, list):
        return None
    hidden = set()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        square = cell.get("square")
        if square is None:
            continue
        visible = cell.get("visible")
        occluded = cell.get("occluded_or_estimated")
        if visible is False or occluded is True:
            hidden.add(square)
    return hidden


def _filter_dets(dets, hidden):
    if not isinstance(dets, list):
        return dets
    if hidden is None:
        return []
    out = []
    for det in dets:
        if isinstance(det, dict):
            square = det.get("square")
            if square is not None and square in hidden:
                continue
        out.append(det)
    return out


def mask_view(view, grid):
    new_view = copy.deepcopy(view)
    if not isinstance(new_view, dict):
        return new_view

    if not isinstance(grid, dict) or not grid.get("valid"):
        new_view["grid_valid"] = False
        new_view["large"] = []
        new_view["raw"] = []
        new_view["cells"] = {}
        new_view["aux_visibility"] = {}
        return new_view

    hidden = _hidden_squares(grid)
    grid_valid = True
    if hidden is None:
        grid_valid = False
        hidden = set()

    new_view["grid_valid"] = bool(new_view.get("grid_valid", True)) and grid_valid

    new_view["large"] = _filter_dets(new_view.get("large"), hidden)
    new_view["raw"] = _filter_dets(new_view.get("raw"), hidden)

    aux = {}
    cells = new_view.get("cells")
    if isinstance(cells, dict):
        for key in list(cells.keys()):
            if key in hidden:
                aux[key] = False
                del cells[key]
            else:
                aux[key] = True
    new_view["aux_visibility"] = aux
    return new_view


class GridBridge(weighted.GridBridge):
    def _remember(self, cam, seq, captured_at, grid_valid, grid_meta,
                  dets_canonical, stats, grid_elapsed, process_ms):
        with self._lock:
            super()._remember(cam, seq, captured_at, grid_valid, grid_meta,
                              dets_canonical, stats, grid_elapsed, process_ms)
            if cam != 0:
                return
            if not self._views.get(0):
                return
            grid = None
            try:
                grid = self.locators[0].last_grid
            except Exception:
                grid = None
            self._views[0] = mask_view(self._views[0], grid)

    def _render(self, display, cam, grid_disp, dets_disp, grid_valid):
        if cam == 1:
            return super()._render(display, cam, grid_disp, dets_disp, grid_valid)

        hidden = None
        if isinstance(grid_disp, dict) and isinstance(grid_disp.get("cells"), list):
            hidden = _hidden_squares(grid_disp)

        filtered_dets = dets_disp
        if hidden is not None:
            filtered_dets = _filter_dets(dets_disp, hidden)

        result = super()._render(display, cam, grid_disp, filtered_dets, grid_valid)

        if hidden is None or not grid_valid:
            return result

        for cell in grid_disp.get("cells", []):
            if not isinstance(cell, dict):
                continue
            square = cell.get("square")
            if square is None or square not in hidden:
                continue
            quad = cell.get("quad")
            center = cell.get("center")
            if quad is not None:
                try:
                    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
                    cv2.polylines(result, [pts], True, (0, 140, 255), 2)
                except Exception:
                    pass
            if center is not None:
                try:
                    cx, cy = int(center[0]), int(center[1])
                    label = "%s ?" % str(square)
                    cv2.putText(result, label, (cx - 18, cy - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 140, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

        try:
            n_hidden = len(hidden)
            text = "C1 visible %d | occluded/outside %d: unknown; use C2" % (
                max(0, len(grid_disp.get("cells", [])) - n_hidden), n_hidden)
            cv2.putText(result, text, (8, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        return result

    def status(self, now=None):
        result = super().status(now)
        grid = None
        try:
            grid = self.locators[0].last_grid
        except Exception:
            grid = None

        aux = {}
        if isinstance(grid, dict) and isinstance(grid.get("cells"), list):
            hidden = _hidden_squares(grid)
            if hidden is not None:
                for cell in grid.get("cells", []):
                    if isinstance(cell, dict) and cell.get("square") is not None:
                        aux[cell["square"]] = cell["square"] not in hidden

        if isinstance(result, dict):
            result["auxiliary_visibility"] = aux
            method = None
            reason = None
            if isinstance(grid, dict):
                method = grid.get("method")
                reason = grid.get("reason")
            result["auxiliary_grid_method"] = method
            result["auxiliary_grid_reason"] = reason
            result["auxiliary_visibility_description"] = (
                "C1 mask uses a fixed recorded gripper area, not dynamic segmentation")
        return result


def main():
    import data_collection_preview_v14 as app
    app.GridBridge = GridBridge
    app._GRID_BRIDGE_IMPORT_ERROR = None
    sys.exit(app.main())


if __name__ == "__main__":
    main()

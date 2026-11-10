# Current ten-step success video

[Watch on the project page](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/#successful-game) · [Download MP4](https://wangyh666-ust.github.io/Chess-Player------SOARM-101/assets/current-success-seed202-20261110.mp4)

This video shows the successful **second session, seed 202**, using the current scanned-piece models and native-contact control. All ten original actions completed pickup, transfer, supported release, settling and checked return home in one persistent physical scene.

| Item | Value |
| --- | --- |
| Source outcome | 10 / 10 complete actions |
| Original simulation | 80,283 ticks at 240 Hz; 334.5125 seconds |
| Displayed recorded states | 1,133 |
| MP4 | 1,148 frames, 12 fps, 1280 × 540 |
| Duration | 95.667 seconds |
| Playback | Approximately 3.5× average; variable speed at phase boundaries |
| Final pause | 15 frames |
| New manipulation or physics steps | 0 |

Every phase's first and last state and all ten checked-home endings are retained, alongside regular half-second samples. The video resets a separate display world to the measured arm joints, gripper angle and all twelve piece transforms. It does not interpolate a planned route or rerun grasp dynamics. Original piece/robot mesh geometry is retained; neutral white/charcoal presentation colors are used.

The first serial display export was stopped after measured throughput forecast an overrun of its 900-second limit. Its 352-frame encoded prefix was retained. Three disjoint suffix ranges were rendered in parallel with the same reviewed renderer, then concatenated. Intermediate renderer end pauses were removed; the final 15-frame pause was retained. This was display processing, with native stepping prohibited throughout.

The selected recorded transforms and action labels were checked against their original trace rows in a separate pass by the same GPT (`independent=false`). Media decode and range coverage are recorded in [verification.json](verification.json); source hashes, selection and rendering provenance are in [provenance.json](provenance.json), [selection.json](selection.json) and [selected-poses.json.gz](selected-poses.json.gz).

This is one successful session from the current three-session assessment, whose complete-action results remain **0/10, 10/10, 5/10**. The video does not establish all-layout or hardware reliability. See [the control handoff](../../CONTROL_HANDOFF.md) for current limits and remaining issues.

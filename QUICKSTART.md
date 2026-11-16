# Automatic shared vision state

Run the calibrated camera as usual. JSON now updates after every analysed frame,
with the highest-confidence valid segmented puff inside the workspace. No keypress
is needed. `--json-out target.json` keeps the existing path; without that option a
calibrated camera writes `vision_state.json`.

The file contains `frame_id`, Unix `timestamp`, `state`, and `target` with `x_mm`,
`y_mm`, `theta_deg`, `width_mm`, and `confidence`. No valid puff (or normal program
exit) writes `target: null`. Updates replace the file atomically.

The robot's main program reads a fresh snapshot when ready to pick, runs its action,
then reads a new snapshot. Frame/request IDs are observations, not robot commands.
Reject stale timestamps (e.g. older than 2 seconds) in case vision crashes or stops.

# AUKEY: preview, calibrate, export

## Lens curvature correction (once per camera/settings)

Print `assets/checkerboard.svg` at actual size (landscape), keeping its white margin.
It has **10x7 squares / 9x6 internal corners**. Keep the print flat on stiff backing.
Then run:

```bash
./AUKEY.command lens
```

Show the entire checkerboard to the AUKEY. Move it to different parts of the view
and **tilt it in different directions**; keep the camera itself fixed. Whenever
the coloured corners appear, press **Space** to save a view. Capture at least
**12 distinct views**, then press **Enter** to fit and save `lens.json`.
If the fit reports poor quality or insufficient tilt, capture better views or use
`r` to restart. `q` cancels. Do not change camera focus, zoom, or resolution afterward.

Then redo `./AUKEY.command calibrate`, and use `./AUKEY.command run` as usual.
The program now corrects radial/tangential curvature **before** rotating the image,
calibrating the paper, detecting objects, or exporting JSON. The HUD says
`Lens correction ON`. Frame dimensions are preserved; no automatic crop is applied.
The paper calibration is tied to the measured lens parameters: changing/removing
`lens.json` requires a new paper calibration, preventing old clicks from being
used on a different corrected view. Until `lens.json` exists, the camera stays raw.

Windows uses the same correction file format and CLI:
`uv run --extra ml leukerbad-detect --camera 0 --lens-calibrate`.
Use the actual demo camera/settings, then calibrate the paper on that setup.

This follows OpenCV's camera-calibration method:
https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

## Paper calibration and manual JSON export

Run these from the repository root, one at a time. Quit the camera with `q`
before starting another command. The launcher selects AUKEY PC-W1 by name.

```bash
./AUKEY.command preview
./AUKEY.command calibrate
./AUKEY.command run
```

Your current view has the **385 mm side horizontally** and **280 mm side vertically**:
calibration uses **A→B = 385 mm; A→D = 280 mm**.
Mark a rectangle A (top left), B (top right), C (bottom right), D (bottom left).
Measure between mark centres. Keep the paper flat and fixed relative to the camera.
The AUKEY launcher initially turns the image 180 degrees. In calibration, press
**t** until the robot is at the bottom. Rotation resets any existing clicks.
Then click **A, B, C, D, and the physical centre/origin mark**, then press Enter.
Choose A at the top-left of the rotated view, then the other corners clockwise.
A 90/270-degree turn swaps the 385/280 mm width and height automatically;
the calibration window shows the A-B and A-D dimensions for verification.
`r` resets the clicks. The centre mark becomes `(0,0)`, with **+X toward the top**
(away from the robot) and **+Y horizontally to the left**. The mark can be slightly off-centre:
the actual clicked mark defines zero. Its location and the chosen rotation are saved.
Recalibrate old four-point files to enable the new centred frame.
This saves `workspace.json`; repeat if the camera, paper, zoom,
crop, orientation, or capture resolution changes.

In `run` mode, press **j** to write **one** calibrated target to `target.json`.
It chooses the valid segmented object with smallest paper Y, then X. It exports
no target without a mask, or when the mask crosses the calibrated rectangle.
Press **s** for an annotated snapshot and **q** to quit.

The live view shows a border around the calibrated paper, dims everything outside
it by 50%, and projects a dot every **10 mm (1 cm)** along both paper axes. The
red origin and +X/+Y arrows identify the coordinate frame. Puff labels in this
mode use calibrated millimetres and paper-plane angles, matching the JSON.
The dots are a screen overlay, not a physical printout; compare them with ruler
marks on the actual paper to check the mapping.

**Click any spot inside the paper** to show a magenta marker and its calibrated
`x, y` position in millimetres at the bottom of the screen and in the terminal.
The last clicked spot stays visible until the next click. Clicks outside the paper
report that they are outside the workspace. Clicking only measures the spot;
pressing `j` remains the separate action that exports a detected puff to JSON.

The JSON includes a unique `request_id`, capture request creation time,
`coordinate_frame: "paper_mm"`, `state: "ready"`, and a `target` with
`x_mm`, `y_mm`, `theta_deg`, `width_mm`, and `confidence`.
If there is no eligible mask, it writes `state: "empty", target: null`.
The JSON also records `origin: "clicked_mark"` and `axes: {"x":"up","y":"left"}`.
Below the origin is negative X; to its right is negative Y. Theta follows the puff's
long axis **anticlockwise from +X (up)**. It is signed around vertical:
**0 degrees = vertical, +45 degrees = leaning left, -45 degrees = leaning right,
90 degrees = horizontal**. The undirected axis repeats after 180 degrees; output
is in `(-90, 90]` so a nearly vertical puff reads close to zero. JSON records this
as `theta_direction: "anticlockwise"`, `theta_period_deg: 180`.
Corner clicks are normalized to the displayed top/left directions while retaining
the measured side lengths. Existing five-point
paper calibrations can be reused for this axis change; restart the program.
The centre is the centre of the mask's oriented enclosing rectangle.

This is a **manual handoff**: wait for the robot to finish and clear the view,
then press j again for a fresh target. Automatic robot acknowledgement is not
implemented. A repeated j can select the same object if it was not removed.
Your friend's program must consume each request ID once and ignore old files.
Paper coordinates still need alignment with the robot's coordinate system.
Calibration maps the paper plane; check physical accuracy using known marks and
account for the height of the vape above the paper before using robot motion.

On Windows use the Python CLI instead of `.command`:

```powershell
uv sync --extra ml
uv run --extra ml leukerbad-detect --camera 0 --calibrate 385 280 --calibration workspace.json
uv run --extra ml leukerbad-detect --camera 0 --weights models/vape_seg.pt --device cpu --conf 0.60 --calibration workspace.json --json-out target.json
```

Calibrate on the actual demo camera and PC; do not transfer the Mac calibration
to a different view. Windows hardware has not been tested here.

## Windows JSON sharing

`vision_state.json` is generated locally and must not be committed. If updating
from a version that tracked it, stop vision and the robot, move that file to
`vision_state.local-backup.json`, then run `git pull`. Vision recreates it on start.

The writer retries temporary access-denied errors while replacing the complete
JSON snapshot. A persistent lock logs a warning and skips that update; the camera
continues. The robot reader closes the file after each read and retries temporary
read errors. Both publisher and robot reader require confidence >= 0.60. Old
timestamps remain invalid after two seconds. Keep consumer reads short, using
`with open(...)`, rather than holding the JSON file open during robot movement.

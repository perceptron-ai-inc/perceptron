# Tutorial — Isaac 0.1 Frame-by-Frame

Run the Isaac 0.1 model on consecutive frames of `surf.mp4` so you can detect surfers across the entire clip and render the results back into a video.

## Assets
- Video input: `cookbook/_shared/assets/tutorials/isaac_0.1_frame_by_frame/surf.mp4`
- Outputs: `frames/` (raw JPEG frames), `frames_annotated/` (overlay frames), `surf_annotated.mp4`

## Steps
1. Export `PERCEPTRON_API_KEY` with a valid credential.
2. Install dependencies: `pip install --upgrade perceptron opencv-python pillow tqdm`.
3. Run the notebook (`isaac_0.1_frame_by_frame.ipynb`) or the script to:
   - Extract frames from the MP4 (adjust stride for speed vs. fidelity).
   - Call `perceptron.detect()` on each frame targeting surfers/surfboards.
   - Draw bounding boxes per frame and stitch them into an annotated MP4.

> Coordinate outputs use the Perceptron 1–1000 normalized grid, so the helper converts them back to pixels before drawing.


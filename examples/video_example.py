"""Example: Analyze a local video file using the Perceptron SDK.

This example demonstrates how to use the video() node to upload and analyze
a local video file. The video is automatically uploaded via presigned URLs.

Usage:
    export PERCEPTRON_API_KEY=your_api_key
    python examples/video_example.py
"""
import perceptron
from perceptron import perceive, video, text

perceptron.configure(
    provider="perceptron",
    api_key="ak.rB-d-J3XL_tG3gqPSPobxg.ubamDxJT9HXkBk07yu9CqmlHHzIPWKuqODYxgUXGqv0",
)

@perceive(model="qwen3-vl-235b-a22b-thinking")
def describe_video(video_path: str):
    """Describe what happens in a video."""
    return video(video_path) + text("Describe what is happening in this video.")


if __name__ == "__main__":
    # Use a sample video from Downloads
    video_path = "/Users/armenag/Downloads/golf_swing.mp4"

    print(f"Analyzing video: {video_path}")
    result = describe_video(video_path)
    print(f"\nDescription:\n{result}")

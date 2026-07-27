import os
import google.generativeai as genai
import PIL.Image
import time
from skvideo.io import vwrite
import cv2

json_generation_config = {'response_mime_type': 'application/json'}


class GeminiVLM:

    def __init__(self, model='gemini-2.5-pro-preview-03-25', temp_dir=None, json_output=False, system_instruction=None):
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel(model, generation_config=json_generation_config if json_output else None,
            system_instruction=system_instruction)
        if temp_dir is None:
            temp_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    def get_image_from_file(self, image_path):
        return PIL.Image.open(image_path)
    
    def get_image_from_array(self, image_array):
        return PIL.Image.fromarray(image_array)
    
    def get_video_from_file(self, video_path):
        success = False
        while not success:
            video_file = genai.upload_file(path=video_path)
            while video_file.state.name == "PROCESSING":
                # print('Processing video...')
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                print(f'Video failed to upload: {video_file.state.name}. Retrying...')
            else:
                success = True
        return video_file
    
    def get_video_from_array(self, video_array):
        timeid = time.strftime('%m%d%H%M%S%f', time.localtime(time.time()))
        temp_video_path = os.path.join(self.temp_dir, f'temp_video_{timeid}.mp4')
        vwrite(temp_video_path, video_array, outputdict={'-vcodec': 'h264', '-pix_fmt': 'yuv420p'})
        video_file = self.get_video_from_file(temp_video_path)
        os.remove(temp_video_path)
        return video_file
    
    def get_keyframes_from_file(self, video_path, num_frames=5):
        """
        Extracts keyframes from the given video file without uploading it.

        Args:
            video_path (str): Path to the video file.
            num_frames (int): Number of keyframes to extract.

        Returns:
            List[Image.Image]: List of extracted keyframes as PIL images.
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(total_frames // num_frames, 1)

        keyframes = []
        for i in range(num_frames):
            frame_idx = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                pil_image = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                keyframes.append(pil_image)

        cap.release()
        return keyframes
    
    def generate_content(self, query_list, temperature=0, text_output=True):
        success = False
        while not success:
            try:
                print('generating content...')
                response = self.model.generate_content(query_list,
                    generation_config=genai.types.GenerationConfig(temperature=temperature),
                    safety_settings=[
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS",
                            "threshold": "BLOCK_NONE",
                        },
                    ]
                )
                response.resolve()
                print('content generated')
                success = True
            except Exception as e:
                print(f"Gemini retrying... {e}")
                time.sleep(3)
        if text_output:
            return response.text
        else:
            return response


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini VLM")
    parser.add_argument('--model', type=str, default='gemini-2.5-pro-preview-03-25', help="Gemini model to use")
    parser.add_argument('--video-path', type=str)
    args = parser.parse_args()

    gemini = GeminiVLM(model=args.model, temp_dir='./tmp', json_output=True)
    keyframes = gemini.get_keyframes_from_file(args.video_path, num_frames=10)
    prompt = '''
    You are assisting a robot in aligning a grasped part for insertion using visual feedback from a camera mounted on the robot's wrist.

    Task:
    - The part is grasped by the robot and can move in four directions: ["up", "down", "left", "right"], each by 2 mm in the camera frame.
    - The goal is to move the part to align it precisely with the hole for insertion.

    Instructions:
    - Carefully observe the video frames. Focus only on the position of the part relative to the hole.
    - Determine the single best action to move the part to align with the hole.
    - Focus only on spatial cues: Is the part too far left, right, above, or below the hole?

    Response format:
    {
    "action": "right",
    "reason": "The part is too far left relative to the hole and needs to move right to align."
    }

    Only output the single best action based on spatial cues. If the part is already aligned, output "hold".
    What is the best action to move the part to align with the hole?
    '''

    query_list = [prompt] + keyframes
    response = gemini.generate_content(query_list, temperature=0.1)
    print(response)
    
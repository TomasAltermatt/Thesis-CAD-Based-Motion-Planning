import os
from moviepy.editor import VideoFileClip, concatenate_videoclips


def combine_animation(record_dir, output_path, reverse):

    file_extension = output_path.split('.')[-1].lower()
    assert file_extension == 'mp4', 'output file must be an MP4 file'

    file_path_list = []
    for file_name in os.listdir(record_dir):
        if file_name.endswith('.gif') or file_name.endswith('.mp4'):
            file_path = os.path.join(record_dir, file_name)
            file_path_list.append(file_path)
    file_path_list.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
    if reverse:
        file_path_list = file_path_list[::-1]

    # Load the first video to determine resolution and FPS
    first_clip = VideoFileClip(file_path_list[0])
    resolution = first_clip.size  # (width, height)
    fps = first_clip.fps  # frames per second

    # Load and optionally resize all video/GIF files
    clips = [VideoFileClip(file).resize(newsize=resolution) for file in file_path_list]

    # Concatenate all the clips together
    final_clip = concatenate_videoclips(clips)
    final_clip.duration -= 1e-6 # fix numeric division issue

    # Write the combined video to a file
    final_clip.write_videofile(output_path, codec="h264", fps=fps, ffmpeg_params=['-pix_fmt', 'yuv420p'], logger=None)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--record-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--reverse', default=False, action='store_true')
    args = parser.parse_args()

    combine_animation(args.record_dir, args.output_path, args.reverse)
    
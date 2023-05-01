#!/usr/bin/env python3

from annotator import run

import argparse
import asyncio
from helper_utils import AnnotationFileManager

import os
import shutil
import subprocess
from lxml import etree

def prepare(input_folder_name, output_folder_name):
    for folder_name in (input_folder_name, output_folder_name):
        try:
            shutil.rmtree(folder_name)
        except FileNotFoundError:
            pass
        os.mkdir(folder_name)


async def main(
    files_limit, max_n_frames, input_dir='input', output_dir='output', is_local_mode=False,
    annotations_table='AnnotationsSWF'
):
    video_files_manager = AnnotationFileManager(
        annotations_table=annotations_table, files_limit=files_limit, detection_type='base'
    )
    await video_files_manager.download_annotations_data(pano_filter=False)

    main_folder_name = "../main"
    input_folder_name = f"{main_folder_name}/{input_dir}/"
    output_folder_name = f"{main_folder_name}/{output_dir}/"

    with open(f"{main_folder_name}/class_list.txt", 'w') as class_list:
        class_list.write("basketball\nstable camera")

    # print(f"Files num: {len(video_files_manager.files)}")

    for file, _ in video_files_manager.files:
        prepare(input_folder_name, output_folder_name)
        shutil.copyfile(file.LocalVidPath, f'{input_folder_name}/{file.Name}.mp4')

        run(max_n_frames=max_n_frames)

        game_type = 'other'
        stable_camera = False

        res = None
        try:
            cmd = ["grep", "-r", "-F", "<object>", f"{main_folder_name}/{output_dir}/PASCAL_VOC/"]
            res = subprocess.check_output(cmd).decode("utf-8")
        except subprocess.CalledProcessError:
            pass

        if res is not None:
            lines = res.split('\n')
            file_names = [line.split(':')[0] for line in lines]
            for file_name in file_names:
                if not file_name:
                    continue

                tree = etree.parse(file_name)
                root = tree.getroot()
                for element in root:
                    if element.tag != 'object':
                        continue
                    for el in element:
                        if el.tag != 'name':
                            continue
                        if el.text == 'basketball':
                            game_type = 'basketball'
                        if el.text == 'stable camera':
                            stable_camera = True

        if is_local_mode:
            continue
        
        video_files_manager._update_annotations_table(file, game_type=game_type, stable_camera=stable_camera)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate shots')
    parser.add_argument('--max_video_num', default=None, help='max number of videos to annotate')
    parser.add_argument('--is_local_mode', action='store_true', help='local mode - do not update annotations table')
    parser.add_argument('--max_n_frames', default=None, help='max number of frames to show')
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args.max_video_num, args.max_n_frames, is_local_mode=args.is_local_mode))

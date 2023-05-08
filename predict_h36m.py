#!/usr/bin/env python3
import argparse
import functools
import itertools
import glob
import pathlib
import os

import numpy as np
import spacepy
import tensorflow as tf
import imageio

import data.datasets3d
import data.h36m
import options
import paths
import poseviz
import video_io
from options import FLAGS, logger


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,  default='/home/sj/Documents/Models/metrabs_eff2l_y4')
    parser.add_argument('--output-path', type=str, default='/home/sj/Documents/Datasets')
    parser.add_argument('--out-video-dir', type=str, default='/home/sj/Documents/Datasets')
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--frame-step', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--correct-S9', action=options.BoolAction, default=False)
    parser.add_argument('--viz', action=options.BoolAction, default=False)
    parser.add_argument('--extract',type=bool, help='Set True if you have to extract frames from video')
    parser.add_argument('--custom', type=str, help='Set your Custom dataset type (e.g Random_Box, Moving_Box ..')

    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    initialize()
    
    save_dir=f"/home/sj/Documents/Datasets/h36m/{FLAGS.custom}/S9"
    save_path=f'{save_dir}/predictions_h36m.npz'
    
    # metrabs는 영상에서 프레임을 이미지로 extract하여 사용
    if FLAGS.extract:
        extract_frames_customdata()
        
        
    model = tf.saved_model.load(FLAGS.model_path)
    skeleton = 'h36m_17'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    predict_fn = functools.partial(
        model.estimate_poses_batched, internal_batch_size=0, num_aug=FLAGS.num_aug,
        antialias_factor=2, skeleton=skeleton)

    #write_video=bool(FLAGS.out_video_dir),매개변수 지움
    viz = poseviz.PoseViz(
        joint_names, joint_edges, 
        world_up=(0, 0, 1), ground_plane_height=0,
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None

    image_relpaths_all = []
    coords_all = []
    
    
    ## Original h36m dataset (S9, S11)
    if not FLAGS.custom:
        for i_subject in (9,11):
            print("현재 실험 중 디렉토리: ",i_subject)
            for activity_name, camera_id in itertools.product(
                    data.h36m.get_activity_names(i_subject), range(4)):
                # if FLAGS.viz:
                #     viz.new_sequence()
                #     if FLAGS.out_video_dir:
                #         viz.start_new_video(
                #             f'{FLAGS.out_video_dir}/S{i_subject}/{activity_name}.{camera_id}.mp4',
                #             fps=max(50 / FLAGS.frame_step, 2))
                #         print('Video saved at', FLAGS.out_video_dir)

                logger.info(f'Predicting S{i_subject} {activity_name} {camera_id}...')
                frame_relpaths, bboxes, camera = get_sequence(i_subject, activity_name, camera_id)
                frame_paths = [f'{paths.DATA_ROOT}/{p}' for p in frame_relpaths]
                box_ds = tf.data.Dataset.from_tensor_slices(bboxes)
                ds, frame_batches_cpu = video_io.image_files_as_tf_dataset(
                    frame_paths, extra_data=box_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)

                coords3d_pred_world = predict_sequence(predict_fn, ds, frame_batches_cpu, camera, viz)
                image_relpaths_all.append(frame_relpaths)
                coords_all.append(coords3d_pred_world)
        np.savez(
            'predictions_h36m.npz', image_path=np.concatenate(image_relpaths_all, axis=0),
            coords3d_pred_world=np.concatenate(coords_all, axis=0))
        print('np saved...')

    
    
    ## Truncated h366m dataset(S9)
    elif FLAGS.custom:
        i_subject=9
        print("현재 실험 중 디렉토리: ",i_subject)
        for activity_name, camera_id in itertools.product(
                data.h36m.get_activity_names(i_subject), range(4)):
            # if FLAGS.viz:
            #     viz.new_sequence()
            #     if FLAGS.out_video_dir:
            #         viz.start_new_video(
            #             f'{FLAGS.out_video_dir}/S{i_subject}/{activity_name}.{camera_id}.mp4',
            #             fps=max(50 / FLAGS.frame_step, 2))
            #         print('Video saved at', FLAGS.out_video_dir)

            logger.info(f'Predicting S{i_subject} {activity_name} {camera_id}...')
            frame_relpaths, bboxes, camera = get_sequence(i_subject, activity_name, camera_id)
            frame_paths = [f'{paths.DATA_ROOT}/{p}' for p in frame_relpaths] # home/sj/Documents/Datasets/h36m/Random_Box/S9/Images/Walking 1.60457274
            box_ds = tf.data.Dataset.from_tensor_slices(bboxes)
            ds, frame_batches_cpu = video_io.image_files_as_tf_dataset(
                frame_paths, extra_data=box_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)

            coords3d_pred_world = predict_sequence(predict_fn, ds, frame_batches_cpu, camera, viz)
            image_relpaths_all.append(frame_relpaths)
            coords_all.append(coords3d_pred_world)
            
        np.savez(
            save_path, image_path=np.concatenate(image_relpaths_all, axis=0),
            coords3d_pred_world=np.concatenate(coords_all, axis=0))
        print('np saved...')


   
    if FLAGS.viz:
        viz.close()


def predict_sequence(predict_fn, dataset, frame_batches_cpu, camera, viz):
    predict_fn = functools.partial(
        predict_fn, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
        extrinsic_matrix=camera.get_extrinsic_matrix()[np.newaxis],
        distortion_coeffs=camera.get_distortion_coeffs()[np.newaxis],
        world_up_vector=camera.world_up)

    pose_batches = []

    for (frames_b, box_b), frames_b_cpu in zip(dataset, frame_batches_cpu):
        boxes_b = tf.RaggedTensor.from_tensor(box_b[:, tf.newaxis])
        pred = predict_fn(frames_b, boxes_b)
        pred = tf.nest.map_structure(lambda x: tf.squeeze(x, 1).numpy(), pred)
        pose_batches.append(pred['poses3d'])

        if FLAGS.viz:
            for frame, box, pose3d in zip(frames_b_cpu, box_b.numpy(), pred['poses3d']):
                viz.update(frame, box[np.newaxis], pose3d[np.newaxis], camera)

    return np.concatenate(pose_batches, axis=0)


def get_sequence(i_subject, activity_name, i_camera):
    camera_name = ['54138969', '55011271', '58860488', '60457274'][i_camera]
    camera = data.h36m.get_cameras(
        f'{paths.DATA_ROOT}/h36m/Release-v1.2/metadata.xml')[i_camera][i_subject - 1]
    coord_path = (f'{paths.DATA_ROOT}/h36m/S{i_subject}/'
                  f'MyPoseFeatures/D3_Positions/{activity_name}.cdf')

    with spacepy.pycdf.CDF(coord_path) as cdf_file:
        coords_raw_all = np.array(cdf_file['Pose'], np.float32)[0]
    n_total_frames = coords_raw_all.shape[0]
    coords_raw = coords_raw_all[::FLAGS.frame_step]
    i_relevant_joints = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
    world_coords = coords_raw.reshape([coords_raw.shape[0], -1, 3])[:, i_relevant_joints]
    
    if not FLAGS.custom: ## original h36m dataset
        image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'# h36m/S9/Images/Walking 1.60457274
    elif FLAGS.custom: ## Truncated h36m dataset
        image_relfolder = f'h36m/{FLAGS.custom}/S{i_subject}/Images/{activity_name}.{camera_name}'# h36m/Random_Box/S9/Images/Walking 1.60457274

    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'  # h36m/S9/Images/Walking 1.60457274/ + frame_000000.jpg'
                      for i_frame in range(0, n_total_frames, FLAGS.frame_step)]
    bbox_path = f'{paths.DATA_ROOT}/h36m/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes = np.load(bbox_path)[::FLAGS.frame_step]
    assert len(bboxes) == len(image_relpaths) == len(world_coords)
    if FLAGS.correct_S9:
        world_coords = data.h36m.correct_world_coords(world_coords, coord_path)
        bboxes = data.h36m.correct_boxes(bboxes, bbox_path, world_coords, camera)

    return image_relpaths, bboxes, camera





def extract_frames_customdata():
    
    video_paths = sorted(glob.glob(f'{paths.DATA_ROOT}/h36m/{FLAGS.custom}/S9/*.mp4', recursive=True)) ## 여기서 Random_Box 파서로 관리하기
    for video_path in video_paths:
        video_name=pathlib.Path(video_path).stem #_ALL.60457274
        dst_folder_path=pathlib.Path(video_path).parents[1]/'S9'/'Images'/video_name #/home/sj/Documents/Datasets/h36m/Random_Box/Images/_ALL.60457274
        os.makedirs(dst_folder_path,exist_ok=True)
        
        with imageio.get_reader(video_path,'ffmpeg') as reader:
            for i_frame, frame in enumerate(reader):
                if any(i_frame % 1==0):
                    dst_filename = f'frame_{i_frame:06d}.jpg'
                    dst_path = os.path.join(dst_folder_path, dst_filename)
                    print(dst_path)
                    imageio.imwrite(dst_path, frame, quality=95)
        
        
if __name__ == '__main__':
    main()

import argparse
import functools
import itertools
import logging
import os
import os.path

import tensorflow as tf
import numpy as np
import spacepy

import cameralib
import data.datasets3d
import data.h36m
import options
import poseviz
import paths
import video_io
from options import FLAGS,logger


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str,  default='/home/sj/Documents/Models/metrabs_eff2l_y4')
    parser.add_argument('--video-dir', type=str, default='/home/sj/Documents/Datasets/h36m/Random_Box/S9')
    parser.add_argument('--video-filenames', type=str)
    parser.add_argument('--viz-downscale', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-detections', type=int, default=-1)
    parser.add_argument('--out-video-path', type=str)
    parser.add_argument('--write-video', action=options.BoolAction)
    parser.add_argument('--num-aug', type=int, default=5)
    parser.add_argument('--fov', type=float, default=55)
    parser.add_argument('--skeleton', type=str, default='h36m_17')
    parser.add_argument('--model-name', type=str, default='')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--viz', action=options.BoolAction, default=False)
    parser.add_argument('--data', type=str, help='Set your dataset type (e.g TruncationData')
    parser.add_argument('--custom', type=str, help='Set your Custom dataset type (e.g Random_Box, Moving_Box ..')



    options.initialize(parser)
    logging.getLogger('absl').setLevel('ERROR')
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    # 초기 설정
    initialize()
    model = tf.saved_model.load(FLAGS.model_path)
    joint_names = model.per_skeleton_joint_names[FLAGS.skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[FLAGS.skeleton].numpy()
    predict_fn = functools.partial(
        model.detect_poses_batched, default_fov_degrees=FLAGS.fov,
        detector_threshold=0.5, detector_flip_aug=True,
        num_aug=FLAGS.num_aug, detector_nms_iou_threshold=0.8, internal_batch_size=64 * 3,
        skeleton=FLAGS.skeleton, suppress_implausible_poses=True, antialias_factor=2,
        max_detections=FLAGS.max_detections)

    # 저장할 배열 
    image_relpaths_all = []
    coords_all = []
    i_subject=9
    print("Starting Predicting with H36m subject ",i_subject)

    # poseviz 설정
    viz = poseviz.PoseViz(
        joint_names, joint_edges, 
        world_up=(0, 0, 1), ground_plane_height=0,
        queue_size=2 * FLAGS.batch_size) if FLAGS.viz else None
    
    # 저장 장소와 이름
    save_dir=f'/home/sj/Documents/Datasets/h36m/2dKeypointNpz/H36m/{FLAGS.data}/metrabs'
    save_path=f'{save_dir}/NoBBOX_predictions_h36m.npz'
    
    
    print("Starting Predicting with H36m subject ",i_subject)
    for activity_name, camera_id in itertools.product(
            data.h36m.get_activity_names(i_subject), range(4)):
    
            print(f'Predicting S{i_subject} {activity_name} {camera_id}...')
            frame_relpaths,bboxes, camera = get_sequence(i_subject, activity_name, camera_id)
            frame_paths = [f'{paths.DATA_ROOT}/{p}' for p in frame_relpaths]
            box_ds = tf.data.Dataset.from_tensor_slices(bboxes)
            ds, frame_batches_cpu = video_io.image_files_as_tf_dataset(
                frame_paths, extra_data=box_ds, batch_size=FLAGS.batch_size, tee_cpu=FLAGS.viz)
            coords3d_pred_world = predict_sequence(predict_fn, ds, frame_batches_cpu, camera, viz)
            image_relpaths_all.append(frame_relpaths)
            coords_all.append(coords3d_pred_world)
            print("\ncoords_all\n",coords_all)
            print("shape: ",coords_all[0].shape)
            
            

    # np.savez(save_path,image_path=np.concatenate(image_relpaths_all, axis=0),
    #          coords3d_pred_world=np.concatenate(coords_all, axis=0))
    # print('np saved...')

    if FLAGS.viz:
        viz.close()

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
    
    if FLAGS.data=="TruncationData" : # Truncationdata
            image_relfolder = f'h36m/{FLAGS.custom}/S{i_subject}/Images/{activity_name}.{camera_name}'# h36m/Random_Box/S9/Images/Walking 1.60457274
    else: # Normal data
            image_relfolder = f'h36m/S{i_subject}/Images/{activity_name}.{camera_name}'# h36m/Random_Box/S9/Images/Walking 1.60457274

    image_relpaths = [f'{image_relfolder}/frame_{i_frame:06d}.jpg'  # h36m/Random_Box/S9/Images/Directions.54138969/frame_000001.jpg
                      for i_frame in range(0, n_total_frames, FLAGS.frame_step)]
    bbox_path = f'{paths.DATA_ROOT}/h36m/S{i_subject}/BBoxes/{activity_name}.{camera_name}.npy'
    bboxes = np.load(bbox_path)[::FLAGS.frame_step]
    assert len(image_relpaths) == len(world_coords)

    return image_relpaths, bboxes,camera


def predict_sequence(predict_fn,dataset, frame_batches_cpu,camera,viz):
    #metrabs 상에서 
    predict_fn = functools.partial(
    predict_fn, intrinsic_matrix=camera.intrinsic_matrix[np.newaxis],
    extrinsic_matrix=camera.get_extrinsic_matrix()[np.newaxis],
    distortion_coeffs=camera.get_distortion_coeffs()[np.newaxis],
    world_up_vector=camera.world_up)

    pose_batches = []


    for (frames_b, box_b), frames_b_cpu in zip(dataset, frame_batches_cpu):
        # gt bounding box 없이 predict, 자체적으로 bounding box를 탐지함.
        pred = predict_fn(frames_b)
        pred = tf.nest.map_structure(lambda x: x.numpy(), pred)  
        
        pose3d=[]
        for i, a in enumerate(pred['poses3d']):
            if a.shape[0]!=1: # 포즈 없을 때
                print("frame no pose!")
                pose3d.append(np.zeros((1,17,3),dtype=np.float32))
            else: # 포즈 있을 때
                pose3d.append(a)
        pred['poses3d']=np.stack(pose3d)
        
        
        if FLAGS.viz:
            for frame, boxes, poses3d in zip(frames_b_cpu, pred['boxes'], pred['poses3d']):
                viz.update(frame, boxes, poses3d, camera)

        pred['poses3d']=np.squeeze(pred['poses3d'],axis=1)        
        pose_batches.append(pred['poses3d'])
       
                
    return np.concatenate(pose_batches, axis=0)


if __name__ == '__main__':
    main()
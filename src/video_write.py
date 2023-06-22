#!/usr/bin/env python3
import os
import argparse
import sys
import cameralib
import cv2
import tensorflow as tf
import poseviz
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import options
from options import FLAGS, logger



def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', type=bool, default=False,help="Show plt while rendering video")
    parser.add_argument("--origin", type=bool, default=False,help="True if you want to render with original video on left side")
    parser.add_argument("--path", required=True, type=str)
    options.initialize(parser)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    initialize()
    model_path='/home/sj/Documents/Models/metrabs_eff2l_y4'

    model = tf.saved_model.load(model_path)
    skeleton = 'h36m_17'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    # fig와 ax 생성
    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111, projection='3d')

    frame_batches = tf.data.Dataset.from_generator(
        frames_from_video, tf.uint8, [None, None, 3]).batch(32).prefetch(1)
    
    writer = cv2.VideoWriter('skeleton_animation.avi', cv2.VideoWriter_fourcc(*'MJPG'), 50, (1000, 1000))

    for frame_batch in frame_batches:
        pred = model.detect_poses_batched(frame_batch, skeleton=skeleton, default_fov_degrees=55)
        camera = cameralib.Camera.from_fov(55, frame_batch.shape[1:3])
        for frame, boxes, poses3d in zip(
                frame_batch.numpy(), pred['boxes'].numpy(), pred['poses3d'].numpy()):
            if len(poses3d) > 0:
                # 스켈레톤 그리기
                plot_skeleton(poses3d[0], joint_edges, fig, ax)

            else:
                ax.clear()
                ax.text(0.5, 0.5,0.5, 'frame no pose', horizontalalignment='center', verticalalignment='center', fontsize=20) # 텍스트 추가
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_zticklabels([])
                
            if FLAGS.origin:
                ax1 = fig.add_subplot(121)

                # 원래 영상 그리기
                ax1.clear()
                ax1.imshow(frame)
                ax1.axis('off')

            plt.pause(0.01)
            # 영상으로 저장
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img)

    writer.release()
    plt.close
                

def frames_from_video():
    video_path = FLAGS.path
    cap = cv2.VideoCapture(video_path)
    while (frame_bgr := cap.read()[1]) is not None:
        yield frame_bgr[..., ::-1]


def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path


def plot_skeleton(poses3d, joint_edges, fig, ax):
    # clear skeleton
    ax.clear()

    for joint1, joint2 in joint_edges:
        x1, y1, z1 = poses3d[joint1]
        x2, y2, z2 = poses3d[joint2]
        if joint1 in [0,1,2,3,8,13,14,15]:
            ax.plot([x1, x2], [z1, z2],[y1, y2],  color='red')
        else:
            ax.plot([x1, x2], [z1, z2], [y1, y2], color='black')


    center = np.mean(poses3d, axis=0)
    #x, y, z 축 범위 설정
    max_range = np.max(np.abs(poses3d - center))
    ax.set_xlim3d([-max_range + center[0], max_range + center[0]])
    ax.set_ylim3d([-max_range + center[2], max_range + center[2]])
    ax.set_zlim3d([-max_range + center[1], max_range + center[1]])


    # z 축 뒤집기
    ax.invert_zaxis()
    
    # hide axis figs   
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # 카메라 위치 변경 (elev: 수직각도 azim: 수평회전각도) 75 90
    ax.view_init(elev=15, azim=-100)
    

    plt.show(block=False) 

if __name__ == '__main__':
    main()
     

  
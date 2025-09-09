
import cv2
import os
from retinaface import RetinaFace
import tensorflow as tf
from tensorflow.keras.models import load_model

h=299
# 设置 GPU 显存按需增长
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"已设置 {len(gpus)} 个 GPU 的显存按需增长")
    except RuntimeError as e:
        print(e)

def video_file(video_dir, video_start, num_videos):
    all_files = sorted(os.listdir(video_dir))
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    if not video_files:
        print(f"目录 {video_dir} 中没有找到视频文件。")
        return []
    end_index = video_start + num_videos
    selected_videos = video_files[video_start:end_index]
    video_frames = [os.path.join(video_dir, f) for f in selected_videos]
    return video_frames


def detect_and_save_faces(img, image_path, output_folder):
    # 使用RetinaFace检测人脸
    faces = RetinaFace.detect_faces(img)

    if faces is None:
        print(f"在图像 {image_path} 中未检测到人脸")
        return

    # # 添加调试信息
    # print(f"检测到的人脸数据: {faces}")

    try:
        # 获取第一个人脸的键
        first_key = next(iter(faces))
        facial_area = faces[first_key]['facial_area']
    except (StopIteration, KeyError) as e:
        print(f"无法访问第一个人脸的数据: {e}")
        return

    x1, y1, x2, y2 = facial_area

    #计算中心点
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    # 设置人脸范围是299*299
    x1_new = cx - h / 2
    y1_new = cy - h / 2
    x2_new = cx + h / 2
    y2_new = cy + h / 2

    # 求整一下
    x1_new = int(round(x1_new))
    y1_new = int(round(y1_new))
    x2_new = int(round(x2_new))
    y2_new = int(round(y2_new))

    # 限制坐标在图像范围内
    x1_new = max(0, x1_new)
    y1_new = max(0, y1_new)
    x2_new = min(img.shape[1], x2_new)
    y2_new = min(img.shape[0], y2_new)

    # 裁剪并保存人脸区域
    face_img = img[y1_new:y2_new, x1_new:x2_new]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    face_file_name = os.path.join(output_folder, f"{base_name}_face_0.png")
    cv2.imwrite(face_file_name, face_img)
    # print(f"已保存第一张人脸图像为 {face_file_name}")
    # print(relative_face_file_name )


def extract_random_frames(video_path, num_frames, output_folder_video):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频文件：{video_path}")
    print(f"帧率 (FPS)：{fps}")
    print(f"总帧数：{total_frames}")

    if total_frames == 0:
        print(f"视频文件 {video_path} 没有帧。")
        cap.release()
        return

    duration = total_frames / fps if fps > 0 else 0
    print(f"视频时长：{duration} 秒")

    frame_idx = 0
    for i in range(num_frames):
        frame_number = int(i * total_frames / num_frames)
        print(f"正在处理第 {i+1}/{num_frames} 帧，跳转到视频帧索引：{frame_number}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret or frame is None or frame.size == 0:
            print(f"读取第 {frame_number} 帧失败，帧为空。")
            continue

        frame_filename = os.path.join(
            output_folder_video, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_idx}.png"
        )
        frame_idx += 1

        # 提取人脸
        detect_and_save_faces(frame,frame_filename,output_folder_video)


    cap.release()
    print(f"从 {video_path} 抽取了 {frame_idx} 帧并保存到 {output_folder_video}")


import os


def process_videos_in_folders_from_txt(txt_file_path, output_base_folder):
    # 读取txt文件，获取视频文件夹路径和对应的帧数
    with open(txt_file_path, 'r') as file:
        lines = file.readlines()

    # 遍历文件中的每一行
    for line in lines:
        # 每行格式: video_dir num_frames
        video_dir, num_frames = line.strip().split()

        # 检查视频目录是否存在
        if not os.path.exists(video_dir):
            print(f"视频目录 {video_dir} 不存在，跳过。")
            continue

        # 为当前视频目录创建对应的输出文件夹
        relative_path = os.path.relpath(video_dir, r"/data/ssd2/tangshuai/DFIL_original_videos/DFIL_original_videos")  # 输出路径是原始路径
        current_output_folder = os.path.join(output_base_folder, relative_path)

        # 确保输出文件夹存在
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        # 调用函数处理视频文件夹
        print(f"Processing videos in folder: {video_dir} -> Output folder: {current_output_folder}")
        process_videos_in_folders(video_dir, current_output_folder, int(num_frames))


def process_videos_in_folders(video_dir, output_base_folder, num_frames):
    # 遍历视频目录中的所有子文件夹
    for root, dirs, files in os.walk(video_dir):
        if not files:  # 如果当前目录没有文件，跳过
            continue

        # 为当前子文件夹创建对应的输出文件夹
        relative_path = os.path.relpath(root, video_dir)
        current_output_folder = os.path.join(output_base_folder, relative_path)

        # 确保输出文件夹存在
        if not os.path.exists(current_output_folder):
            os.makedirs(current_output_folder)

        # 处理当前子文件夹中的所有视频文件
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):  # 根据需要过滤视频文件类型
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                print(f"Processing video: {video_path} -> Output folder: {current_output_folder}")
                extract_random_frames(video_path, num_frames, current_output_folder)


if __name__ == '__main__':
    # 设置txt文件路径和输出路径
    txt_file_path = r"/data/ssd2/tangshuai/DFIL_original_videos/List_of_testing_videos.txt"
    output_folder = r"/data/ssd2/tangshuai/DFIL299_new"

    # 调用函数从txt文件中读取并处理视频文件夹
    process_videos_in_folders_from_txt(txt_file_path, output_folder)
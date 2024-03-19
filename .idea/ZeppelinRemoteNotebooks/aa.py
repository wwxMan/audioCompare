from fastapi import FastAPI, Request, Form
# 导入需要的库
import librosa
import numpy as np
from numpy.linalg import norm
from numpy import array
from scipy.spatial import distance
from scipy import signal
import os
import io
from loguru import logger
import heapq
from cachetools import LRUCache, cached
import concurrent.futures
from functools import partial
import math
from dtw import dtw
from pydub import AudioSegment
from pydub.silence import split_on_silence
from fastdtw import fastdtw
import time

# 创建一个缓存对象 LRUCache 基于最近最少使用的缓存策略
infinite_cache = LRUCache(maxsize=500)
infinite_cache2 = LRUCache(maxsize=500)
infinite_cache3 = LRUCache(maxsize=500)
infinite_cache4 = LRUCache(maxsize=500)
infinite_cache5 = LRUCache(maxsize=500)
infinite_cache6 = LRUCache(maxsize=10000)
# 存储模板的节拍
infinite_cache7 = LRUCache(maxsize=500)
infinite_cache8 = LRUCache(maxsize=10000)

# 默认采样率
default_sr = 3200
default_dbfs = -25
default_dtw_pass = 0.75
default_silence_min_duration = 400
default_silence_threshold = -40

app = FastAPI()

# 日志操作
logger.add('logs/log.log', rotation='1 days', compression='zip', enqueue=True,
           format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
           retention='90 days', encoding='utf8', level="INFO")
logger.add('logs/warn.log', rotation='1 days', compression='zip', enqueue=True,
           format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>',
           retention='90 days', encoding='utf8', level="WARNING")

fibonacci_array = [2, 3, 5, 8, 13, 8, 5, 3, 2]


def fibonacci(n):
    return fibonacci_array[n]


@app.post('/add')
def add_num(num1: str = Form(...), num2: str = Form(...)):
    # 使用Request对象获取POST请求的参数
    result = int(num1) + int(num2)
    return str(my_method(int(num1)))


# 定义一个任务函数
def task(num, num2):
    # logger.info('num:{}'.format(num))
    # logger.info('num2:{}'.format(num2))
    return num * 2 + num2


def my_method(args):
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 使用partial函数传递额外的参数
        partial_task = partial(task, num2=3)
        # 提交任务给线程池
        results = [executor.submit(partial_task, arg) for arg in range(args)]

        # 收集任务的结果
        output = [result.result() for result in concurrent.futures.as_completed(results)]

    # 根据需求进行其他操作
    # ...

    # 返回结果（如果需要）
    return output


@app.post('/contains')
def contains(targetFile: str = Form(...), templateFile: str = Form(...)):
    # 获取总时长
    target_total_duration = librosa.get_duration(filename=targetFile)
    # 计算起始时间点
    target_start_time = target_total_duration - duration

    # 加载音频文件并提取特征
    y1, sr1 = librosa.load(targetFile)
    y2, sr2 = librosa.load(compareFile)
    # 计算相关性
    correlation = np.correlate(y1, y2)
    return str(np.max(correlation))


# 模板音频可能出现的时间点
@app.post('/audioMayAppear')
def audio_may_appear(targetFile: str = Form(...), compareFile: str = Form(...)):
    # 需要读取的时长（单位：秒）
    duration = int(request.form.get('duration'))
    offset = find_offset(targetFile, compareFile, duration)
    return (f"模板音频最可能出现的时间点: {offset}s")


# 模板音频可能出现的时间点
@app.post('/splitAudio')
def split_audio(ringFile: str = Form(...), type: str = Form(...)):
    target_file = ringFile
    type = type
    segments = 0
    # logger.info(f'target_file:{target_file}')
    # 判断比较的路径是否为文件夹
    if os.path.isdir(target_file):
        # 遍历文件夹中的文件
        for filename in os.listdir(target_file):
            # 构建文件的完整路径
            file_path = os.path.join(target_file, filename)
            if os.path.isfile(file_path):
                segments += spilt_audio_to_file(file_path, os.path.splitext(os.path.basename(file_path))[0], type)
    else:
        segments = spilt_audio_to_file(target_file, os.path.splitext(os.path.basename(target_file))[0], type)

    return str(f"模板音频共拆分: {segments}份，文件保存相对路径: output/{type}")

@app.post('/beatCompare')
def beat_compare(ringFile: str = Form(...), templateFile: str = Form(...)):
    target_file = ringFile
    compare_file = templateFile
    compare_result = {}
    # 判断比较的路径是否为文件夹
    if os.path.isdir(compare_file):
        # 遍历文件夹中的文件
        for filename in os.listdir(compare_file):
            # 构建文件的完整路径
            file_path = os.path.join(compare_file, filename)
            if os.path.isfile(file_path):
                # 获取模板音频文件的总时长
                compare_total_duration = get_compare_duration_from_cache(file_path)
                target_total_duration = get_compare_duration_from_cache(target_file)
                compare_beat = get_beat_from_cache(file_path, compare_total_duration)
                target_beat = get_beat_from_cache(target_file, target_total_duration)
                # dist, cost, acc, path = dtw(target_beat, compare_beat, dist=lambda target_beat, compare_beat: norm(target_beat - compare_beat, ord=1))
                dist, cost, acc, path = dtw(compare_beat, target_beat,
                                            dist=lambda compare_beat, target_beat: norm(compare_beat - target_beat,
                                                                                        ord=1))
                compare_result[filename] = dist
                # logger.info('dist:{}'.format(dist))
    # logger.info('compare_result:{}'.format(str(compare_result)))
    matched_song = min(compare_result, key=compare_result.get)
    return (matched_song)


# @app.post('/splitAudioCompare')
# def split_audio_compare(ringFile: str = Form(...), templateFile: str = Form(...), tops: str = Form(...)):
#     target_file = request.form.get('ringFile')
#     compare_file = request.form.get('templateFile')
#     # 需要的最多结果
#     tops = request.form.get('tops')
#
#     # 获取目标音频切片
#     target_array = get_spilt_target_audio_from_cache(target_file, os.path.splitext(os.path.basename(target_file))[0],
#                                                      "target")
#     # logger.info(f'target_array:{len(target_array)}')
#
#     result_array = []
#     stop = False
#     # 判断比较的路径是否为文件夹
#     if os.path.isdir(compare_file):
#         # 遍历文件夹中的文件
#         for filename in os.listdir(compare_file):
#             if stop:
#                 break
#             # 构建文件的完整路径
#             file_path = os.path.join(compare_file, filename)
#             if os.path.isfile(file_path):
#                 #
#                 compare_array = get_spilt_audio_from_cache(file_path, os.path.splitext(os.path.basename(file_path))[0],
#                                                            "compare")
#                 # logger.info(f'compare_array:{len(compare_array)}')
#                 for i,compare_audio in compare_array:
#                     if stop:
#                         break
#                     compare_result = []
#                     compare_result_array = []
#                     for j,target_audio in target_array:
#                         similar = stat_dtw_similar(compare_audio,  target_audio, default_sr)
#                         # logger.info(f'dist:{similar}')
#                         compare_result.append(similar)
#                         if similar > default_dtw_pass:
#                             stop = True
#                             break
#                     compare_result_array.append(filename)
#                     # logger.info('compare_result:{}'.format(str(compare_result)))
#                     compare_result_array.append(max(compare_result))
#                     result_array.append(compare_result_array)
#     else:
#         compare_array = get_spilt_audio_from_cache(compare_file, os.path.splitext(os.path.basename(compare_file))[0],
#                                                    "compare")
#         # logger.info(f'compare_array:{len(compare_array)}')
#         for compare_audio in compare_array:
#             if stop:
#                 break
#             compare_result = []
#             compare_result_array = []
#             for target_audio in target_array:
#                 similar = stat_dtw_similar(compare_audio, default_sr, target_audio, default_sr)
#                 # logger.info(f'dist:{similar}')
#                 compare_result.append(similar)
#                 if similar > default_dtw_pass:
#                     stop = True
#                     break
#             compare_result_array.append(os.path.splitext(os.path.basename(compare_file))[0])
#             # logger.info('compare_result:{}'.format(str(compare_result)))
#             compare_result_array.append(max(compare_result))
#             result_array.append(compare_result_array)
#
#     if result_array is not None:
#         if tops:
#             result_array = heapq.nlargest(int(tops), result_array, key=lambda s: s[1])
#         else:
#             result_array = heapq.nlargest(400, result_array, key=lambda s: s[1])
#     return str(result_array)


@app.post('/initCache')
def init_cache(templateFile: str = Form(...)):
    initCache(templateFile)


def initCache(compare_file):
    # 判断比较的路径是否为文件夹
    if os.path.isdir(compare_file):
        # 遍历文件夹中的文件
        for filename in os.listdir(compare_file):
            # 构建文件的完整路径
            file_path = os.path.join(compare_file, filename)
            if os.path.isfile(file_path):
                get_spilt_audio_from_cache(file_path, os.path.splitext(os.path.basename(file_path))[0],
                                           "compare")


def stat_dtw_audio(audio_1, sr_1, audio_2, sr_2):
    # logger.info(f'audio_1:{len(audio_1)}')
    # logger.info(f'audio_2:{len(audio_2)}')
    length = min(len(audio_1), len(audio_2))
    # 相差
    differ = abs(len(audio_1) - len(audio_2))
    mfcc_1 = librosa.feature.mfcc(y=audio_1, sr=sr_1)
    mfcc_2 = librosa.feature.mfcc(y=audio_2, sr=sr_2)
    dtw_distance, path = fastdtw(mfcc_1.T, mfcc_2.T, dist=distance.euclidean)

    return dtw_distance


# 匹配度超过0.5 会进行额外的dtw可信度计算 可信度小于default_dtw_pass 匹配度不可信，将匹配度/2 返回
def stat_dtw_similar(compare_audio, target_audio, compare_correlations, target_correlations, compare_index,
                     target_index):
    compare_correlation = compare_correlations[compare_index]
    target_correlation = target_correlations[target_index]

    relative_correlation = signal.correlate(compare_audio, target_audio, mode='valid', method='fft')
    max_correlate_compare = np.max(compare_correlation)
    max_correlate_target = np.max(target_correlation)
    max_correlate_relative = np.max(relative_correlation)

    reliability = max_correlate_relative / max_correlate_target
    similarity = max_correlate_relative / max_correlate_compare
    logger.info(f'ori similarity:{similarity}')
    rate = max_correlate_target / max_correlate_compare
    # 计算可靠的相似度
    # similarity = get_reliable_result(rate, reliability, similarity)
    similarity = get_reliable_result(rate, reliability, similarity)
    logger.info(f'compare_index-{compare_index},target_index-{target_index},ori similarity:{similarity}')
    length = min(len(compare_audio), len(target_audio))
    max_length = max(len(compare_audio), len(target_audio))
    if similarity < 0.15:
        return similarity
    logger.info(f'reliability:{reliability}')
    logger.info(f'stat similarity:{similarity}')
    length = min(len(compare_audio), len(target_audio))
    logger.info('length scalar:{}'.format(max_length / length))
    # 相差
    # differ = abs(len(audio_1) - len(audio_2))
    # 判断数组长度并调整大小
    compare_audio_sp = compare_audio[:length]
    target_audio_sp = target_audio[:length]
    mfcc_1 = librosa.feature.mfcc(y=compare_audio_sp, sr=default_sr)
    mfcc_2 = librosa.feature.mfcc(y=target_audio_sp, sr=default_sr)
    # 计算音频的 STFT
    # stft1 = np.abs(librosa.stft(compare_audio_sp))
    # stft2 = np.abs(librosa.stft(target_audio_sp))
    # 计算相似性（欧氏距离）
    # euclidean_distance1 = distance.euclidean(stft1.flatten(), stft2.flatten())
    # logger.info('euclidean_distance1:{}'.format( euclidean_distance1))
    #
    dtw_distance, path = fastdtw(mfcc_1.T, mfcc_2.T, dist=distance.euclidean)
    # euclidean_distance = distance.euclidean(mfcc_1.flatten(), mfcc_2.flatten())
    # logger.info('length:{}'.format(length))
    # logger.info('euclidean_distance:{}'.format( euclidean_distance))
    # # 使用曼哈顿距离计算相似度
    # manhattan_distance = distance.cityblock(mfcc_1.flatten(), mfcc_2.flatten())
    # logger.info('manhattan_distance:{}'.format(manhattan_distance))

    dtw_similar_front = 1 - dtw_distance / length
    min_dtw_similar = dtw_similar_front
    logger.info(f'dtw_similar_front:{dtw_similar_front}')
    # 如果正比没比到就反比下 正面反面
    if dtw_similar_front < default_dtw_pass:
        compare_audio_sp = compare_audio[len(compare_audio) - length:]
        target_audio_sp = target_audio[len(target_audio) - length:]
        mfcc_1 = librosa.feature.mfcc(y=compare_audio_sp, sr=default_sr)
        mfcc_2 = librosa.feature.mfcc(y=target_audio_sp, sr=default_sr)
        dtw_distance, path = fastdtw(mfcc_1.T, mfcc_2.T, dist=distance.euclidean)
        dtw_similar_back = 1 - dtw_distance / length
        logger.info(f'dtw_similar_back:{dtw_similar_back}')
        min_dtw_similar = max(dtw_similar_front, dtw_similar_back)
    logger.info(f'min_dtw_similar:{min_dtw_similar}')
    if min_dtw_similar >= default_dtw_pass:
        similarity += 0.6
    else:
        similarity /= 3
    logger.info(f'final_similar:{similarity}')
    if similarity > 1:
        return 1
    else:
        return similarity


def find_offset(within_file, find_file, duration):
    y_within, sr_within = librosa.load(within_file, sr=None)
    y_find, _ = librosa.load(find_file, sr=sr_within)
    y_find, _ = librosa.load(find_file, sr=sr_within)
    c = signal.correlate(y_within, y_find[:sr_within * duration], mode='valid', method='fft')
    peak = np.argmax(c)
    offset = round(peak / sr_within, 2)
    return offset


def find_offset_start_from_duration(target_audio, compare_audio, compare_sr, target_total_duration, duration):
    target_start_time = target_total_duration - duration
    if target_start_time < 0:
        target_start_time = 0
    c = signal.correlate(target_audio[int(compare_sr * target_start_time):], compare_audio, mode='valid', method='fft')
    peak = np.argmax(c)
    # 数组下标除以采样率获得时间 单位秒
    offset = round(peak / compare_sr, 2)
    if target_start_time > 0:
        offset += target_start_time
    return offset


@app.post('/audioCompareBySeconds')
def audio_compare_by_seconds(ringFile: str = Form(...), templateFile: str = Form(...),
                             tops: str = Form(...), duration: str = Form(...), passValue: str = Form(...),
                             type: int = Form(...)):
    # 需要校验的音频文件
    target_file = ringFile
    compare_file = templateFile
    # 需要读取的时长（单位：秒）
    duration = int(duration)
    # 可以通过的阀值
    pass_value = passValue
    # 需要的最多结果
    tops = tops

    logger.info('targetFile:{}'.format(target_file))
    logger.info('templateFile:{}'.format(compare_file))
    logger.info('duration:{}'.format(duration))
    if pass_value and not isinstance(pass_value, str):
        return {'error': 'Invalid optional parameter'}, 400

    if pass_value:
        logger.info('passValue:{}'.format(pass_value))
        pass_value = float(pass_value)
    # 结果集
    result_array = []
    # 判断音频是否存在
    if not os.path.exists(target_file):
        return str(result_array)
    # 获取音频文件的大小（字节数） 小于2048 直接丢弃
    file_size = os.path.getsize(target_file)
    if file_size < 2048:
        return str(result_array)

    # 判断比较的路径是否为文件夹
    if os.path.isdir(compare_file):
        # 使用线程池
        # 遍历文件夹中的文件
        # result_array = check_similar_async(compare_file, duration, target_file, pass_value)
        for filename in os.listdir(compare_file):
            # 构建文件的完整路径
            file_path = os.path.join(compare_file, filename)
            # 检查文件是否为普通文件
            if os.path.isfile(file_path):
                if type and type == 2:
                    # 在这里进行文件处理操作
                    part_array = check_similar(file_path, duration, target_file, pass_value, type)
                else:
                    # 在这里进行文件处理操作
                    part_array = check_similar(file_path, duration, target_file, pass_value, type)
                if pass_value:
                    if part_array[1] >= pass_value:
                        result_array = []
                        result_array.append(part_array)
                        break
                result_array.append(part_array)

    else:
        # logger.info('{}只是一个文件'.format(compare_file))
        part_array = check_similar(compare_file, duration, target_file, pass_value, type)
        result_array.append(part_array)

    if result_array is not None:
        if tops:
            result_array = heapq.nlargest(int(tops), result_array, key=lambda s: s[1])
        else:
            result_array = heapq.nlargest(400, result_array, key=lambda s: s[1])
        if pass_value:
            if result_array[0][1] < pass_value:
                logger.warning(f'文件匹配率过低: {target_file}')

    return str(result_array)


def check_similar_async(compareFile, duration, targetFile, pass_value):
    # 创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # 使用partial函数传递额外的参数
        partial_task = partial(check_similar, duration=duration, target_file=targetFile, pass_value=pass_value)
        # 提交任务给线程池
        results = [executor.submit(partial_task, os.path.join(compareFile, file_path)) for file_path in
                   os.listdir(compareFile)]
        # 收集任务的结果
        output = [result.result() for result in concurrent.futures.as_completed(results)]
    return output


# 从缓存中加载文件
@cached(infinite_cache)
def load_audio_from_cache(compare_file, compare_total_duration):
    # 在这里执行昂贵的操作
    # ...
    # 加载模板音频
    compare_audio, compare_sr = librosa.load(compare_file, offset=0, duration=compare_total_duration, sr=default_sr)
    # logger.info('compare_sr:{}'.format(compare_sr))
    return compare_audio, compare_sr


@cached(infinite_cache2)
def get_compare_duration_from_cache(file_name):
    # 在这里执行昂贵的操作
    # 获取模板音频文件的总时长
    duration = librosa.get_duration(path=file_name)
    return duration


@cached(infinite_cache3)
def load_correlation_from_cache(file_name, compare_total_duration):
    # 在这里执行昂贵的操作
    # 获取模板音频文件的总时长
    compare_audio, compare_sr = load_audio_from_cache(file_name, compare_total_duration)
    total_correlation = signal.correlate(compare_audio, compare_audio, mode='valid', method='fft')
    return total_correlation


# 从缓存中加载文件
@cached(infinite_cache4)
def load_target_audio_from_cache(target_file, compare_sr):
    target_audio, target_sr = librosa.load(target_file, sr=compare_sr)
    return target_audio, target_sr


# 从缓存中加载文件时长
@cached(infinite_cache5)
def get_target_duration_from_cache(target_file):
    target_total_duration = librosa.get_duration(path=target_file)
    return target_total_duration


# 从缓存中加载文件切片
@cached(infinite_cache6)
def get_spilt_audio_from_cache(compare_file, file_name, type):
    return get_spilt_audio(compare_file, file_name, type)


@cached(infinite_cache7)
def get_spilt_target_audio_from_cache(file, file_name, type, duration):
    return get_spilt_audio(file, file_name, type, duration)


# 从缓存中加载模板节拍
@cached(infinite_cache7)
def get_beat_from_cache(file_name, compare_total_duration):
    y, sr = load_audio_from_cache(file_name, compare_total_duration)
    return get_beat_from_audio(y, sr)


# 从缓存中加载文件切片
@cached(infinite_cache8)
def get_segment_audio_from_cache(compare_file, file_name, type):
    return get_segment_audio(compare_file, file_name, type)


def get_beat_from_audio(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_frames = librosa.feature.delta(beat_frames, mode='nearest')
    return array(beat_frames).reshape(-1, 1)

    # def spilt_audio_to_file(audio,sr,file_name):
    #     onset_frames = librosa.onset.onset_detect(y=audio, sr=sr )
    #     #logger.info(f'onset_frames:{onset_frames}')
    #
    #     onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    #     segments = []
    #     start_time = 0.0
    #     #logger.info(f'onset_times:{onset_times}')
    #
    #     for onset_time in onset_times:
    #         segment = audio[int(start_time * sr):int(onset_time * sr)]
    #         segments.append(segment)
    #         start_time = onset_time
    #     for i, segment in enumerate(segments):
    #         output_file = f"output/{file_name}/segment_{i}.wav"  # 设置输出文件名
    #         output_dir = f"output/{file_name}"  # 设置输出文件名
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         with wave.open(output_file, 'w') as wf:
    #             wf.setnchannels(2)  # 设置声道数为双声道
    #             wf.setsampwidth(2)  # 设置样本宽度为2字节（16位）
    #             wf.setframerate(sr)  # 设置采样率
    #             wf.writeframes(segment.tobytes())  # 将音频数据写入文件

    return segments


def normalize_audio(audio):
    # 获取音频的rms值
    dbfs = audio.dBFS

    # 计算音频的增益值
    gain = default_dbfs - dbfs

    # 应用增益值来归一化音频
    normalized_audio = audio.apply_gain(gain)

    return normalized_audio


def spilt_audio_to_file(target_file, file_name, type):
    # 记录程序开始时间
    start_time = time.time()
    audio = AudioSegment.from_file(target_file, format='mp3')
    logger.info("from_file程序执行耗时: {:.2f} 秒".format(time.time() - start_time))
    # 获取音频数据并进行归一化
    audio = normalize_audio(audio)
    output_dir = f"output/{type}"  # 设置输出文件名
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # audio.export(f"output/{type}/{file_name}.mp3", format='mp3')

    logger.info("normalize_audio程序执行耗时: {:.2f} 秒".format(time.time() - start_time))
    # 提取音频的节拍信息
    tempo, beat_frames = librosa.beat.beat_track(y=np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0,
                                                 sr=audio.frame_rate)
    max_beat_frame = max(beat_frames)
    min_beat_frame = min(beat_frames)
    logger.info("beat_frames程序执行耗时: {:.2f} 秒".format(time.time() - start_time))
    logger.info(f"平均节拍（BPM）:{tempo}")
    logger.info(f"节拍（max(beat_frames)）:{max_beat_frame}")
    logger.info(f"节拍（min(beat_frames)）:{min_beat_frame}")
    logger.info(f'dbfs_value:{audio.dBFS}')
    logger.info(f'max_dBFS:{audio.max_dBFS}')

    silence_min_duration = default_silence_min_duration - tempo / 2
    # 如果节拍的中值大于2倍平均值 说明这段话有频率变快的地方 时间需再缩短
    # 如果节拍过小
    if tempo < 100:
        silence_min_duration = 200
    elif (max_beat_frame + min_beat_frame) / 4 > tempo:
        silence_min_duration -= tempo
    elif tempo / min_beat_frame > 3:
        silence_min_duration -= tempo
    elif max_beat_frame / min_beat_frame > 3:
        silence_min_duration -= tempo

    logger.info(f"silence_min_duration: {silence_min_duration} ")
    filtered_segments, segments = filter_seg(audio, int(silence_min_duration), audio.max_dBFS,max_beat_frame, file_name)
    logger.info("filtered_segments程序执行耗时: {:.2f} 秒".format(time.time() - start_time))
    for i, segment in enumerate(filtered_segments):
        output_file = f"output/{type}/{file_name}_{i}.mp3"  # 设置输出文件名
        segment.export(output_file, format='mp3')
    logger.info("for程序执行耗时: {:.2f} 秒".format(time.time() - start_time))
    for i, segment in enumerate(segments):
        output_file = f"output/{type}/{file_name}_se_{i}.mp3"  # 设置输出文件名
        output_dir = f"output/{type}"  # 设置输出文件名
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        segment.export(output_file, format='mp3')

    # logger.info(f"模板音频:{file_name}已被拆分: {len(filtered_segments)}份")
    return len(filtered_segments)


def get_segment_audio(file, file_name, type, duration=None):
    array = []
    correlations = []
    audio_librosa, sr = librosa.load(file, sr=default_sr)
    array.append(audio_librosa)
    # 计算下correlate后存入缓存
    correlations.append(signal.correlate(audio_librosa, audio_librosa, mode='valid', method='fft'))
    return array, correlations


def get_spilt_audio(file, file_name, type, duration=None):
    # 记录程序开始时间
    begin_time = time.time()
    audio = AudioSegment.from_file(file, format='mp3')
    logger.info("audio程序执行耗时: {:.2f} 秒".format(time.time() - begin_time))

    if duration:
        # 获取音频的总时长（毫秒）
        total = len(audio)
        # 计算需要截取的起始时间（毫秒）
        start_time = total - duration * 1000
        # 截取后几秒的音频
        if start_time > 0:
            audio = audio[start_time:]
    # 获取音频数据并进行归一化
    audio = normalize_audio(audio)
    logger.info("normalize_audio程序执行耗时: {:.2f} 秒".format(time.time() - begin_time))

    # 提取音频的节拍信息
    tempo, beat_frames = librosa.beat.beat_track(y=np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0,
                                                 sr=audio.frame_rate)
    if len(beat_frames) == 0:
        max_beat_frame = 10
        min_beat_frame = 10
    else:
        max_beat_frame = max(beat_frames)
        min_beat_frame = min(beat_frames)
    logger.info("beat_track程序执行耗时: {:.2f} 秒".format(time.time() - begin_time))

    silence_min_duration = default_silence_min_duration - tempo / 2
    # 如果节拍的中值大于2倍平均值 说明这段话有频率变快的地方 时间需再缩短
    # 如果节拍过小
    if tempo < 100:
        silence_min_duration = 200
    elif (max_beat_frame + min_beat_frame) / 4 > tempo:
        silence_min_duration -= tempo
    elif tempo / min_beat_frame > 3:
        silence_min_duration -= tempo
    elif max_beat_frame / min_beat_frame > 3:
        silence_min_duration -= tempo


    # silence_min_duration = default_silence_min_duration - tempo
    # if max_beat_frame < 200:
    #     silence_min_duration -= tempo / 4
    # elif max_beat_frame > 320:
    #     silence_min_duration += tempo
    # elif (max_beat_frame + min_beat_frame) / 4 > tempo:
    #     silence_min_duration -= tempo / 2
    # elif max_beat_frame / min_beat_frame > 10:
    #     silence_min_duration += tempo
    # else:
    #     silence_min_duration += tempo / 2
    #
    # if silence_min_duration < 200:
    #     silence_min_duration = 200
    #
    # if silence_min_duration > 600:
    #     silence_min_duration = 600

    filtered_segments, segments = filter_seg(audio, int(silence_min_duration), audio.max_dBFS,max_beat_frame)
    logger.info("filter_seg程序执行耗时: {:.2f} 秒".format(time.time() - begin_time))
    array = []
    correlations = []
    for i, segment in enumerate(filtered_segments):
        output_file = f"output/{type}/{file_name}_{i}.mp3"  # 设置输出文件名
        output_dir = f"output/{type}"  # 设置输出文件名
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        segment.export(output_file, format='mp3')
        # 将音频片段转换为字节流
        # audio_bytes = segment.export(format='mp3').read()
        # 使用librosa加载字节流
        # audio_librosa, sr = librosa.load(y=np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0, sr=default_sr)
        # 将音频片段转换为字节流
        audio_bytes = segment.export(format='mp3').read()
        # 使用librosa加载字节流
        audio_librosa, sr = librosa.load(io.BytesIO(audio_bytes), sr=default_sr)
        logger.info("librosa.load程序执行耗时: {:.2f} 秒".format(time.time() - begin_time))
        # 计算第二个音频的均方根（RMS）能量
        # rms2 = np.sqrt(np.mean(audio_librosa ** 2))
        # logger.info(f'rms2:{rms2}')
        array.append(audio_librosa)
        # 计算下correlate后存入缓存
        correlations.append(signal.correlate(audio_librosa, audio_librosa, mode='valid', method='fft'))

    return array, correlations


def filter_seg(audio, silence_min_duration=400, diff_db=-5,max_beat=None, file_name=None):
    # 静默阈值（以分贝为单位）
    # silence_threshold =default_silence_threshold
    silence_threshold = int(default_silence_threshold + (diff_db / 2 if diff_db > -7 else 0))
    # silence_threshold = default_silence_threshold + diff_db / 2
    logger.info(f'silence_threshold:{silence_threshold}')
    #说的很快 应该要增大截取静默时间
    if max_beat:
        silence_min_duration + max_beat/4
    if silence_min_duration > 550 :
        silence_min_duration = 550

    # 静默持续时间（毫秒）
    # silence_min_duration = 400
    filter_size = 1400
    count = 0
    filtered_segments = []
    segments = []
    # 默认至少能拆出2段
    while (len(filtered_segments) < 2 and silence_threshold > -60):
        filtered_segments, segments = get_filter_seg(audio, silence_threshold, silence_min_duration, filter_size)
        silence_threshold -= fibonacci(count)
        silence_min_duration += fibonacci(count) * 3
        filter_size -= fibonacci(count) * 10
        count += 1
    logger.info(f'silence_threshold:{silence_threshold}')
    logger.info(f'silence_min_duration:{silence_min_duration}')
    logger.info(f'filter_size:{filter_size}')
    logger.info(f'len(segments):{len(segments)}')
    logger.info(f'len(segments):{len(min(segments, key=len))}')

    if len(filtered_segments) == 0:
        logger.warning(f'{file_name}拆分模板失败')

    return filtered_segments, segments


def get_filter_seg(audio, silence_threshold, silence_min_duration, filter_size=1500):
    segments = split_on_silence(audio, min_silence_len=silence_min_duration, silence_thresh=silence_threshold,
                                keep_silence=False, seek_step=3)
    # min_segment_len = len(min(segments, key=len))
    # filter_size = min_segment_len if filter_size - min_segment_len < 300 else filter_size
    # filter_size = max(1050,filter_size)
    logger.info(f"模板音频原始拆分: {len(segments)}份")
    logger.info(f"silence_threshold: {silence_threshold}")
    logger.info(f"silence_min_duration: {silence_min_duration}")
    logger.info(f"filter_size: {filter_size}")
    for segment in segments:
        logger.info(f"len segment: {len(segment)}")

    # 放弃长度小于1.5秒的录音片段 放弃长度过大的
    filtered_segments = [segment for segment in segments if
                         (len(segment) >= filter_size)  and (len(segment) < 10000)]
    if len(filtered_segments) >=2:
        filtered_segments.extend([segment for segment in segments if
                                  (len(segment) < filter_size)  and (filter_size - len(segment)<300) and (len(segment) < 10000)])

    return filtered_segments, segments


def check_similar(compare_file, duration, target_file, pass_value, type=None):
    # logger.info('{} isfile:{}'.format(compare_file, os.path.isfile(compare_file)))

    if not os.path.isfile(compare_file):
        return
    # 返回数组对象 文件名 相似度
    array = []
    array.append(os.path.splitext(os.path.basename(compare_file))[0])
    trust, trust_similar = can_be_trust_similar(compare_file, target_file, duration, type)
    array.append(trust_similar)
    return array
    # 获取音频文件的总时长
    target_total_duration = get_target_duration_from_cache(target_file)
    # logger.info('target_total_duration:{}'.format(target_total_duration))
    # 获取模板音频文件的总时长
    compare_total_duration = get_compare_duration_from_cache(compare_file)
    # logger.info('compare_total_duration:{}'.format(compare_total_duration))

    # 加载模板音频
    # compare_audio, compare_sr = librosa.load(compareFile, offset=0, duration=compare_total_duration)
    compare_audio, compare_sr = load_audio_from_cache(compare_file, compare_total_duration)
    # 所有特征点数
    total_correlation = load_correlation_from_cache(compare_file, compare_total_duration)

    # logger.info('correlation of compare with compare:{}'.format(np.max(total_correlation)))
    # 加载目标文件
    target_audio, target_sr = load_target_audio_from_cache(target_file, compare_sr)

    offset = find_offset_start_from_duration(target_audio, compare_audio, compare_sr, target_total_duration, duration)
    # logger.info(f"模板音频最可能出现的时间点: {offset}s")
    # 可通过阀值
    pass_threshold = 0.7
    if pass_value:
        pass_threshold = pass_value
    # 找到两段音频最大的相关点的时间 然后先截取一次，如果此次结果小与阀值 走后续的方法
    temp_similarity = stat_highest_match_similarity(compare_audio, compare_sr, compare_total_duration,
                                                    target_total_duration, offset,
                                                    target_audio, total_correlation, pass_threshold)
    # logger.info('temp_similarity:{}'.format(temp_similarity))

    if temp_similarity > pass_threshold:
        if temp_similarity > 1:
            temp_similarity = 1
        if temp_similarity > 0.5:
            if not can_be_trust_similar(compare_file, target_file):
                temp_similarity /= 2
        elif temp_similarity > 0.3:
            if can_be_trust_similar(compare_file, target_file):
                temp_similarity += 0.3
        array.append(os.path.splitext(os.path.basename(compare_file))[0])
        array.append(round(temp_similarity, 3))
        return array

    # 截取数组 而不是再读文件
    max_similarity = loop_stat_similarity(compare_audio, compare_sr, compare_total_duration, duration, target_audio,
                                          target_total_duration, total_correlation, pass_threshold)

    if max_similarity > 1:
        max_similarity = 1
    # 如果相似度大于0.5 并且结果不可信 相似度除2
    if max_similarity > 0.5:
        if not can_be_trust_similar(compare_file, target_file):
            max_similarity /= 2
    elif max_similarity > 0.3:
        if can_be_trust_similar(compare_file, target_file):
            max_similarity += 0.3
    array.append(os.path.splitext(os.path.basename(compare_file))[0])
    array.append(round(max(max_similarity, temp_similarity), 3))
    logger.info('array:{}'.format(array))

    return array


# 计算匹配度最大的一段音频的相似度
def stat_highest_match_similarity(compare_audio, compare_sr, compare_total_duration, target_total_duration,
                                  offset, target_audio,
                                  total_correlation, pass_threshold):
    similarity = get_reliable_similarity(compare_audio, compare_sr, compare_total_duration, target_total_duration,
                                         offset, target_audio,
                                         total_correlation, pass_threshold)
    return similarity


def get_reliable_similarity(compare_audio, compare_sr, compare_total_duration, target_total_duration, offset,
                            target_audio,
                            total_correlation, pass_threshold):
    # 截取目标声音
    # 使用librosa加载音频文件的最后几秒
    # 如果目标声音文件根本就没这么长 直接返回 0

    if target_total_duration < compare_total_duration or target_total_duration < offset:
        return 0

    start_time = offset - compare_total_duration
    if start_time < 0:
        start_time = 0
    # logger.info('start_time :{}'.format(start_time))

    # 因为是取2倍 要校验下越界问题
    # 1 开始时间

    if len(target_audio) > start_time * compare_sr + compare_total_duration * compare_sr:
        target_audio_part = target_audio[
                            int(start_time * compare_sr):int(
                                (start_time * compare_sr + compare_total_duration * compare_sr))]
    else:
        target_audio_part = target_audio[int(start_time * compare_sr):]

    correlation = signal.correlate(target_audio_part, compare_audio, mode='valid', method='fft')
    similarity = np.max(correlation) / np.max(total_correlation)
    logger.info('ori similarity:{}'.format(similarity))
    # 可信度 暂时隐藏可信度
    # if similarity > pass_threshold:
    target_correlation = signal.correlate(target_audio_part, target_audio_part, mode='valid', method='fft')
    # logger.info(
    #     'get_reliable_similarity:correlation of target_audio_part with target_audio_part:{}'.format(
    #         np.max(target_correlation)))
    reliability = np.max(correlation) / np.max(target_correlation)
    logger.info('reliability:{}'.format(reliability))
    rate = similarity / reliability
    # 计算可靠的相似度
    similarity = get_reliable_result(rate, reliability, similarity)

    logger.info('stat similarity:{}'.format(similarity))
    return similarity


def can_be_trust_similar(compare_file, target_file, duration, type=None):
    start_time = time.time()
    # 获取目标音频切片
    target_array, target_correlations = get_spilt_target_audio_from_cache(target_file, os.path.splitext(
        os.path.basename(target_file))[0], "target", duration)
    compare_array = None
    compare_correlations = None
    if type and type == 2:
        # 获取模板音频切片
        compare_array, compare_correlations = get_segment_audio_from_cache(compare_file,
                                                                           os.path.splitext(
                                                                               os.path.basename(compare_file))[
                                                                               0], "compare")
    else:
        # 获取模板音频切片
        compare_array, compare_correlations = get_spilt_audio_from_cache(compare_file,
                                                                         os.path.splitext(
                                                                             os.path.basename(compare_file))[
                                                                             0], "compare")
    # 相信
    trust = False
    compare_result = []
    logger.info(f'target_array:{len(target_array)}')
    logger.info(f'compare_array:{len(compare_array)}')
    logger.info(f'compare_file:{compare_file}')

    for i, compare_audio in enumerate(compare_array):
        if trust:
            break
        for j, target_audio in enumerate(target_array):
            similar = stat_dtw_similar(compare_audio, target_audio, compare_correlations, target_correlations, i, j)
            compare_result.append(round(similar, 3))
            if similar > default_dtw_pass:
                trust = True
                break
    logger.info(f'can_be_trust:{trust}')
    return trust, max(compare_result) if len(compare_result) > 0 else 0


# 循环音频文件切片 获取最大相似度
def loop_stat_similarity(compare_audio, compare_sr, compare_total_duration, duration, target_audio,
                         target_total_duration,
                         total_correlation, pass_threshold):
    # 计算起始时间点
    target_start_time = target_total_duration - duration
    # 截取次数
    total_split = duration / compare_total_duration
    # 如果音频时长小于截取时长
    if target_start_time < 0:
        total_split = target_total_duration / compare_total_duration
    # logger.info('total_split:{}'.format(total_split))
    # 循环比较得分 获取最大得分
    count = 1
    max_similarity_1 = 0.0
    # #logger.info('------------loop1----------------')
    # max_similarity_1 = stat_max_similarity(compare_audio, compare_sr, compare_total_duration*2, count, duration,
    #                                        max_similarity_1, target_audio,
    #                                        target_total_duration-compare_total_duration/2, total_correlation, total_split/2,pass_threshold)
    # logger.info('------------loop2----------------')
    # 循环第二次 切分比例为模板音频的2倍 比较得分 获取最大得分
    count_2 = 1
    max_similarity_2 = 0.0

    max_similarity_2 = stat_max_similarity(compare_audio, compare_sr, compare_total_duration * 2, count_2, duration,
                                           max_similarity_2, target_audio,
                                           target_total_duration, total_correlation, total_split / 2, pass_threshold)

    # logger.info('max_similarity1:{}'.format(max_similarity_1))
    # logger.info('max_similarity_2:{}'.format(max_similarity_2))
    max_similarity = max(max_similarity_1, max_similarity_2)

    return max_similarity


# 工作线程执行方法
def worker(compare_audio, compare_sr, total_split, target_total_duration, total_correlation, shared_list):
    shared_list.append(index)


# 计算最大相似
def stat_max_similarity(compare_audio, compare_sr, compare_total_duration, count, duration, max_similarity,
                        target_audio,
                        target_total_duration, total_correlation, total_split, pass_threshold):
    while count < (total_split + 1):
        start_time = target_total_duration - compare_total_duration * count
        if start_time < 0:
            start_time = 0
        # logger.info(f'target_start_time:{start_time}')
        # 加载部分数组
        if (len(target_audio) > start_time * compare_sr + compare_total_duration * compare_sr):
            target_audio_sp = target_audio[
                              int(start_time * compare_sr):int(
                                  start_time * compare_sr + compare_total_duration * compare_sr)]
        else:
            target_audio_sp = target_audio[int(start_time * compare_sr):]

        correlation = signal.correlate(target_audio_sp, compare_audio, mode='valid', method='fft')
        # logger.info('correlation of target with compare:{}'.format(np.max(correlation)))
        similarity = np.max(correlation) / np.max(total_correlation)
        logger.info('ori similarity:{}'.format(similarity))
        # 可信度
        # 可信度 暂时隐藏可信度
        # if similarity > pass_threshold:
        target_correlation = signal.correlate(target_audio_sp, target_audio_sp, mode='valid', method='fft')
        # logger.info('correlation of target with target:{}'.format(np.max(target_correlation)))
        reliability = np.max(correlation) / np.max(target_correlation)

        logger.info('reliability:{}'.format(reliability))
        rate = similarity / reliability
        # 计算可靠的相似度
        similarity = get_reliable_result(rate, reliability, similarity)
        logger.info('stat similarity:{}'.format(similarity))
        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > pass_threshold:
                break

        else:
            if count == 1:
                start_time = compare_total_duration - duration
                if start_time < 0:
                    start_time = 0

                # logger.info(f'target_start_time:{start_time}')
                # 取出部分数组
                target_audio_sp = target_audio[int(start_time * compare_sr):]
                correlation = signal.correlate(target_audio_sp, compare_audio, mode='valid', method='fft')
                # logger.info('correlation of target with compare:{}'.format(np.max(correlation)))
                max_similarity = np.max(correlation) / np.max(total_correlation)
                # logger.info('ori similarity:{}'.format(max_similarity))
                # 可信度
                # 可信度 暂时隐藏可信度
                # if max_similarity > pass_threshold:
                target_correlation = signal.correlate(target_audio_sp, target_audio_sp, mode='valid', method='fft')
                # logger.info('correlation of target with target:{}'.format(np.max(target_correlation)))
                reliability = np.max(correlation) / np.max(target_correlation)
                logger.info('reliability:{}'.format(reliability))
                rate = max_similarity / reliability
                # 计算可靠的相似度
                max_similarity = get_reliable_result(rate, reliability, max_similarity)
                logger.info('stat similarity:{}'.format(max_similarity))
        count += 1

    return max_similarity


def get_reliable_result(rate, reliability, similarity):
    if rate > 1.618 and similarity > 1:
        similarity = similarity / 4 + reliability
        return similarity
    if rate > 1.618 and (similarity - reliability) > 0.25 and reliability < 0.3:
        if similarity > 0.8:
            similarity = similarity / 2 + reliability
        elif similarity > 0.7:
            similarity = reliability * 2
        elif similarity > 0.6:
            similarity = reliability + reliability / (math.ceil(rate))
        elif similarity <= 0.6:
            similarity = reliability + reliability / (math.ceil(rate) + 1)

    return similarity


def audio_cosine(target_audio, target_sr, compare_audio, compare_sr):
    # 加载音频文件并提取特征
    target_audio = librosa.effects.preemphasis(target_audio)
    target_audio = librosa.util.normalize(target_audio)
    # target_audio = librosa.effects.trim(target_audio)
    compare_audio = librosa.effects.preemphasis(compare_audio)
    compare_audio = librosa.util.normalize(compare_audio)
    # compare_audio = librosa.effects.trim(compare_audio)

    mfcc1 = librosa.feature.mfcc(y=target_audio, sr=target_sr)
    mfcc2 = librosa.feature.mfcc(y=compare_audio, sr=compare_sr)
    # 判断数组长度并调整大小
    if mfcc1.shape[1] > mfcc2.shape[1]:
        mfcc1 = np.resize(mfcc1, (mfcc2.shape[0], mfcc2.shape[1]))
    elif mfcc1.shape[1] < mfcc2.shape[1]:
        mfcc2 = np.resize(mfcc2, (mfcc1.shape[0], mfcc1.shape[1]))

    # 使用余弦相似度计算相似度
    # similarity = 1 - distance.cosine(mfcc1.flatten(), mfcc2.flatten())
    # return similarity
    # scaler = StandardScaler()
    # mfcc1_normalized = scaler.fit_transform(mfcc1.T)
    # mfcc2_normalized = scaler.transform(mfcc2.T)
    # similarity = cosine_similarity(mfcc1_normalized, mfcc2_normalized)
    # #logger.info('归一化处理后相似度:{}'.format(similarity[0][0]))

    # 判断数组长度并调整大小
    # 使用欧几里得距离计算相似度
    euclidean_distance = distance.euclidean(mfcc1.flatten(), mfcc2.flatten())
    # similarity = 600 / (1 + euclidean_distance)
    # logger.info('euclidean_distance:{}'.format(euclidean_distance))
    # # 使用曼哈顿距离计算相似度
    manhattan_distance = distance.cityblock(mfcc1.flatten(), mfcc2.flatten())
    # logger.info('manhattan_distance:{}'.format(manhattan_distance))
    # # 使用余弦相似度计算相似度
    similarity = 1 - distance.cosine(mfcc1.flatten(), mfcc2.flatten())
    return similarity


# initCache("/home/guest/audioCompare/music_retrieve/music_base")
# server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
# server.serve_forever()
# if __name__ == '__main__':
#     app.run()

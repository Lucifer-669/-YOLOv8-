# detection_app.py
import cv2
import time
import numpy as np
import gradio as gr
import pygame
from typing import Iterator
from ultralytics import YOLO

# ====================== 系统参数配置 ======================
TRAINED_MODEL = "xxx"  # 替换为你的模型路径
TARGET_CATEGORY = 0     # 要检测的目标类别ID，我这里只是实例（填阿拉伯数字）
SCORE_THRESHOLD = 0.5  # 检测置信度下限
REQUIRED_FRAMES = 15   # 最小连续检测帧数（防误报）
DEFAULT_WEBCAM = 0     # 默认摄像头编号
WARNING_SOUND = "xxx"  # 警报音效路径

# ========================================================

class SafetyMonitor:
    def __init__(self):
        # 初始化YOLOv8检测模型
        self.detector = YOLO(TRAINED_MODEL)
        self.positive_count = 0
        self.warning_active = False
        self.warning_time = 0
        self.warning_limit = 10  # 警报最长持续时间（秒）

        # 设置音频系统
        pygame.mixer.init()
        try:
            self.alert_sound = pygame.mixer.Sound(WARNING_SOUND)
            self.audio_ready = True
        except Exception as e:
            print(f"音频加载失败: {WARNING_SOUND} - {str(e)}")
            self.audio_ready = False

    def evaluate(self, img_data: np.ndarray) -> tuple:
        """分析图像并返回检测结果"""
        detection = self.detector(img_data, verbose=False)[0]
        incident_detected = False
        marked_image = detection.plot()

        # 遍历检测结果
        for item in detection.boxes:
            if int(item.cls) == TARGET_CATEGORY and float(item.conf) > SCORE_THRESHOLD:
                self.positive_count += 1
                if self.positive_count >= REQUIRED_FRAMES:
                    incident_detected = True
                    if not self.warning_active:
                        self.activate_warning()
                    self.warning_active = True
                    self.warning_time = time.time()
                    self.positive_count = 0
                break
        else:
            self.positive_count = max(0, self.positive_count - 1)

        # 检查警报超时
        if self.warning_active and (time.time() - self.warning_time) > self.warning_limit:
            self.warning_active = False
            pygame.mixer.stop()

        return self.warning_active, marked_image

    def activate_warning(self):
        """启动警报系统"""
        if self.audio_ready:
            try:
                self.alert_sound.play(loops=-1)
            except Exception as e:
                print(f"音频播放失败: {str(e)}")


# 初始化监控系统
monitor = SafetyMonitor()


# ==================== 界面功能模块 ====================
def build_warning_panel(active: bool) -> str:
    """生成警报提示面板"""
    if active:
        return """
        <div style='
            color: white;
            background: #ff4444;
            font-size: 24px;
            border: 3px solid #cc0000;
            padding: 15px;
            border-radius: 8px;
            animation: warning_flash 1s infinite;
        '>
         !!! 安全警报：检测到危险行为 !!!
        </div>
        <style>
            @keyframes warning_flash {
                0% {opacity: 1;}
                50% {opacity: 0.7;}
                100% {opacity: 1;}
            }
        </style>
        """
    return ""


def handle_image(input_img: np.ndarray) -> tuple:
    """处理静态图片"""
    converted = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    alert_state, output_img = monitor.evaluate(converted)
    return cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), build_warning_panel(alert_state)


def analyze_video(video_source: str) -> Iterator[tuple]:
    """处理视频流"""
    capture = cv2.VideoCapture(video_source)
    while capture.isOpened():
        ret, current_frame = capture.read()
        if not ret: break

        alert_state, processed = monitor.evaluate(current_frame)
        yield cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), build_warning_panel(alert_state)
    capture.release()


def live_monitoring(cam_id: int) -> Iterator[tuple]:
    """实时监控摄像头"""
    video_stream = cv2.VideoCapture(cam_id)
    if not video_stream.isOpened():
        yield None, "<div style='color: red'>摄像头初始化失败，请检查设备连接！</div>"
        return

    while video_stream.isOpened():
        ret, live_frame = video_stream.read()
        if not ret: break

        alert_state, processed_frame = monitor.evaluate(live_frame)
        yield cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), build_warning_panel(alert_state)
    video_stream.release()


# ==================== 用户界面设计 ====================
with gr.Blocks(theme=gr.themes.Soft(), title="基于YOLOv8的摔倒检测系统") as app_ui:
    gr.Markdown("## 基于YOLOv8的摔倒检测系统")
    gr.Markdown("功能：支持图片分析、视频检测和实时监控")

    # 设备配置区域
    with gr.Row():
        camera_selector = gr.Number(
            label="选择摄像设备",
            value=DEFAULT_WEBCAM,
            precision=0,
            minimum=0,
            maximum=10,
            step=1,
            interactive=True
        )
        gr.HTML("""
        <div style="color: #666; margin-top: 8px">
        设备编号指南：
        <ul>
            <li>0 → 主摄像头</li>
            <li>1 → 外接设备1</li>
            <li>2 → 外接设备2</li>
        </ul>
        </div>
        """)

    # 功能标签页
    with gr.Tabs():
        # 图片分析功能
        with gr.Tab("图片分析"):
            with gr.Row():
                upload_image = gr.Image(label="上传图片", type="numpy")
                result_image = gr.Image(label="分析结果", interactive=False)
            alert_panel_img = gr.HTML()
            analyze_btn = gr.Button("执行分析", variant="primary")
            analyze_btn.click(handle_image, [upload_image], [result_image, alert_panel_img])

        # 视频检测功能
        with gr.Tab("视频检测"):
            with gr.Row():
                video_upload = gr.Video(label="上传视频", sources=["upload"])
                video_output = gr.Image(label="检测结果", streaming=True)
            alert_panel_vid = gr.HTML()
            video_btn = gr.Button("开始检测", variant="primary")
            video_btn.click(analyze_video, [video_upload], [video_output, alert_panel_vid])

        # 实时监控功能
        with gr.Tab("实时监控"):
            with gr.Row():
                live_view = gr.Image(label="监控画面", streaming=True)
                alert_panel_live = gr.HTML()
            monitor_btn = gr.Button("启动监控", variant="primary")
            monitor_btn.click(
                fn=live_monitoring,
                inputs=[camera_selector],
                outputs=[live_view, alert_panel_live],
                show_progress="hidden"
            )

# 启动应用程序
if __name__ == "__main__":
    app_ui.launch(
        server_port=7860,
        show_error=True,
    )
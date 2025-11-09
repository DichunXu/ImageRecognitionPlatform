# 图像识别平台 (ImageRecognitionPlatform)

简要说明  
这是一个基于 ONNX 的轻量级图像识别与演示平台，支持单张上传识别、批量识别、摄像头实时识别与历史/日志可视化。

快速目录（可点击打开）
- 应用入口与页面：[`app.py`](app.py)（路由示例：[`app.upload_file`](app.py)、[`app.api_batch_infer`](app.py)、[`app.video_feed`](app.py)、[`app.api_camera_diagnose`](app.py)、[`app.api_onnx_providers`](app.py)、[`api_app_logs`](app.py)）
- ONNX 推理：[`onnx_infer.py`](onnx_infer.py)（核心类：[`ONNXYoloDetector`](onnx_infer.py)，方法示例：[`ONNXYoloDetector.predict_and_save`](onnx_infer.py)）
- 数据库与连接：[`db.py`](db.py)（包含 [`build_mysql_url`](db.py)、[`get_session`](db.py) 等），[`connect.py`](connect.py)（包含 [`connect_with_params`](connect.py)），数据库初始化：[`db_init.py`](db_init.py)（包含 [`attempt_database_setup`](db_init.py)）
- 默认模型：[`models/best_9.onnx`](models/best_9.onnx)
- 脚本：[`scripts/run_batch_test.py`](scripts/run_batch_test.py)
- 前端模板：[`templates/index.html`](templates/index.html)、[`templates/batch.html`](templates/batch.html)、[`templates/camera.html`](templates/camera.html)、[`templates/db_visualize.html`](templates/db_visualize.html)、[`templates/logs.html`](templates/logs.html)

快速开始（开发）
1. 环境
   - Python 3.8+
   - 推荐依赖：flask, onnxruntime, opencv-python（摄像头/视频），redis（可选）
   - 安装示例：
     ```sh
     pip install flask onnxruntime opencv-python
     ```
2. 准备模型  
   - 将 ONNX 模型放入 `models/`。默认使用 [`models/best_9.onnx`](models/best_9.onnx)（该模型为茶树病害识别模型）。
3. 运行
   - 在项目根目录运行：
     ```sh
     python app.py
     ```
   - 访问 http://127.0.0.1:5000/

主要页面与 API
- 首页（文件上传与单图识别）：[`app.upload_file`](app.py) -> 页面模板 [`templates/index.html`](templates/index.html)
- 批量识别 API：[`app.api_batch_infer`](app.py)（前端页面 [`templates/batch.html`](templates/batch.html) 也调用该接口）
  - 示例脚本：[`scripts/run_batch_test.py`](scripts/run_batch_test.py)
- 摄像头实时流：[`app.video_feed`](app.py) -> 页面 [`templates/camera.html`](templates/camera.html)
- 摄像头诊断：[`app.api_camera_diagnose`](app.py)
- ONNX providers 查询：[`app.api_onnx_providers`](app.py)
- 日志查看页面：[`app.logs_page`](app.py) -> 模板 [`templates/logs.html`](templates/logs.html)
- 数据库可视化：页面 [`/db`](app.py) -> 模板 [`templates/db_visualize.html`](templates/db_visualize.html)，后端读取历史行：[`app.api_db_rows`](app.py)

模型与推理
- 推理封装位于 [`onnx_infer.py`](onnx_infer.py)，主要类为 [`ONNXYoloDetector`](onnx_infer.py)。常用方法：
  - `predict(img, conf, iou)`：对单帧预测
  - `annotate(img, dets, class_names)`：绘制标注框
  - `predict_and_save(in_path, out_path, conf, iou, class_names)`：对文件批量处理并保存结果

缓存与历史
- 缓存（可选）：通过环境变量 `REDIS_URL` 启用（由 [`app.py`](app.py) 中 `_redis_client` 管理）。
- 历史记录存储：默认为文件 `uploads/history.json`（读取/写入函数在 [`app.py`](app.py) 中），若启用数据库则使用表并通过 [`db.py`](db.py) 提供会话操作。
- 清空历史：路由 [`/history/clear`](app.py)

数据库配置
- 支持通过环境变量 `DATABASE_URL` 或直接参数（环境变量 `DB_DIRECT_HOST`、`DB_DIRECT_USER`、`DB_DIRECT_PASS`、`DB_DIRECT_NAME` 等）进行初始化。初始化逻辑在 [`db_init.attempt_database_setup`](db_init.py) 中。辅助函数：[`connect_with_params`](connect.py)、[`build_mysql_url`](db.py)、[`get_session`](db.py)。

日志与调试
- 应用日志写入 `logs/app.log`（由 [`app._setup_logging`](app.py) 配置）
- 日志查看：页面 [`templates/logs.html`](templates/logs.html)；后端接口：[`api_app_logs`](app.py)

部署与注意事项
- 摄像头功能依赖 OpenCV（`opencv-python`）；若未安装，相应路由会返回提示（代码在 [`app.video_feed`](app.py) 与摄像头枚举 [`list_cameras`](app.py) 中）。
- ONNX 模型加载失败会抛出可读错误（见 [`ONNXYoloDetector.__init__`](onnx_infer.py) 的错误提示），常见情况：文件不是有效 ONNX、导出时 opset/graph 不兼容或文件损坏。
- 批量处理请确保传入目录路径为绝对路径或项目内相对路径（详见 [`app.api_batch_infer`](app.py)）。

常用命令
- 运行开发服务器：
  ```sh
  python app.py
  ```
- 运行批量测试脚本：
  ```sh
  python scripts/run_batch_test.py
  ```

文件索引（快速打开）
- [app.py](app.py) — 主应用与路由（如 [`app.upload_file`](app.py)、[`app.api_batch_infer`](app.py)）
- [onnx_infer.py](onnx_infer.py) — 推理实现（[`ONNXYoloDetector`](onnx_infer.py)）
- [db.py](db.py), [connect.py](connect.py), [db_init.py](db_init.py) — 数据库相关（如 [`build_mysql_url`](db.py)、[`connect_with_params`](connect.py)、[`attempt_database_setup`](db_init.py)）
- [models/best_9.onnx](models/best_9.onnx) — 默认权重
- [scripts/run_batch_test.py](scripts/run_batch_test.py) — 批量接口测试脚本
- 前端模板：[`templates/index.html`](templates/index.html)、[`templates/batch.html`](templates/batch.html)、[`templates/camera.html`](templates/camera.html)、[`templates/db_visualize.html`](templates/db_visualize.html)、[`templates/logs.html`](templates/logs.html)

许可证  
本仓库遵循 MIT 许可证（MIT License）。

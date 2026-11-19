# 断层图像三维重建与分析

在仓库根目录运行以下命令启动图像处理、配准、拼接和区域分析界面。

```powershell
python -m pip install -r tool/code_GUI/requirements.txt
python -m tool.code_GUI
```

也可以继续直接运行 `tool/code_GUI/GUI_3D_tomography.py`，或使用 `python -m tool.code_GUI.GUI_3D_tomography`。各页面在首次打开时加载。使用三维查看时，通过 `requirements-3d.txt` 安装 Napari 及其依赖。

图像处理支持 CLAHE、频域带阻及方向性 Median 残差滤除。拼接支持自动估计、填补黑色缝隙和重叠渐变。批量拼接有多组完整网格时，输出文件从 `_0001` 开始编号。区域分析输入为整数标签图，`0` 表示背景。

- `ui/pages/` 存放各功能页面及其参数控件。`ui/main_window.py` 负责导航，`ui/widgets.py` 和 `ui/tasks.py` 分别提供公共控件和后台任务。
- `processing/` 按功能存放算法和导出流程。`images.py` 提供通用图像读写与几何操作，`large_image_backend.py` 接入同级 `large-image-tool` 的计算模块。
- `assets/icons/` 存放导航图标、标志和 IoU 色标。

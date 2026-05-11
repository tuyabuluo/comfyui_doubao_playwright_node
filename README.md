# ComfyUI Doubao AI Playwright Node

这是一个 ComfyUI 自定义节点：打开一个可见的 Playwright 浏览器窗口，把 0-5 张输入图像和文本发送到豆包 AI 网页对话里，等待回复稳定后，尝试下载本轮生成的 0-4 张图片，并通过 1 个 IMAGE batch 端口和 1 个 STRING 文本端口输出。

## 安装

1. 把 `comfyui_doubao_playwright_node` 放到 `ComfyUI/custom_nodes/` 下。
2. 在 ComfyUI 使用的 Python 环境中安装依赖：

```bash
pip install -r ComfyUI/custom_nodes/comfyui_doubao_playwright_node/requirements.txt
python -m playwright install chromium
```

3. 重启 ComfyUI。

## 使用

- 节点名：`Doubao AI Playwright Chat`
- 分类：`Doubao/Playwright`
- 输入图像端口：`image_1` 到 `image_5`
- 输出图像端口：`images`，有几张图就输出几张图的 batch
- 文本输出端口：`text`

第一次运行会打开浏览器并拉到前台。如果豆包要求登录，请在打开的浏览器里手动登录，然后重新运行节点。浏览器会保持打开，后续运行默认继续当前对话；如果你手动最小化浏览器，节点不会在后续运行时主动把它弹到前台。

## 关键参数

- `website_url`：豆包网页地址，默认 `https://www.doubao.com/chat/`
- `image_save_path`：生成图片保存目录，默认 `output/doubao_playwright`
- `new_conversation`：是否开启新对话，默认关闭；开启时本次运行会尝试点击“新对话”
- `try_hd_download`：默认开启；优先悬停生成图并点击图片下方“下载”，再尝试右侧大图“保存”和右键菜单“下载原图”。这些方式失败后才从 `srcset`、父级链接、`data-original/data-hd/data-large` 等大图候选里抓取图片
- 开启 `try_hd_download` 时，如果最终只拿到网页预览小图，节点会跳过这张图而不是把小图输出到图片端口。
- `browser_channel`：默认 `chrome`，也可以填 `msedge` 或 `chromium`
- `browser_executable_path`：如果要指定本机浏览器程序路径，在这里填写
- `user_data_dir`：浏览器用户数据目录；留空时会在保存目录下创建 `playwright_profile`
- `wait_timeout`：等待回复或图像生成的最长秒数
- `min_output_images`：本轮至少等待几张新图片后才结束，默认 `1`；纯文本对话请设为 `0`
- 如果旧工作流把 `min_output_images` 传成 `0`，但提示词明显是在生图，节点会自动修正为 `1` 并在控制台打印实际等待图片数。
- 如果设置了等待图片，但豆包实际只回复文字，节点会在文本稳定且 25 秒内没有新图后判定为纯文字回复，输出文本并跳过图片下载。
- `empty_image_outputs_as_none`：默认开启；没有图片时 `images` 端口返回空值。若后续节点不支持空值，可关闭它恢复黑图占位。

## 选择器参数

网页结构变化时，可以填写这些 selector 让节点更稳定：

- `input_selector`：对话输入框，例如 `textarea` 或 `[contenteditable="true"]`
- `upload_selector`：上传按钮或 `input[type="file"]`
- `send_button_selector`：发送按钮
- `new_chat_selector`：新对话按钮
- `assistant_message_selector`：AI 回复文本所在元素

留空时节点会使用内置的常见选择器自动寻找。

## 注意

这个节点是网页自动化，不是豆包官方 API。网页改版、登录验证、权限弹窗、下载按钮变化都会影响自动化成功率。节点不会绕过验证码或登录限制；请遵守目标网站的使用条款。

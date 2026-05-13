# ComfyUI Doubao Playwright Node

把豆包网页能力接入 ComfyUI 的自定义节点。节点会启动一个可见的 Playwright Chromium/Chrome/Edge 浏览器，把文本和图片发送到豆包网页对话中，等待回复稳定后返回文本，或下载本轮生成的图片并作为 ComfyUI `IMAGE` 输出。

> 这是网页自动化节点，不是豆包官方 API。它不会绕过登录、验证码、权限弹窗或网站限制。

## 功能

- 支持文生文、文生图、图生文、图文生图 4 类节点。
- 支持最多 5 张输入图片参与图文生图。
- 文生图和图文生图会尝试下载本轮新增图片，最多返回 4 张图片组成的 batch。
- 自动复用浏览器登录状态，首次运行后可在打开的浏览器中手动登录豆包。
- 自动加载 `browser_extension` 下的浏览器扩展，优先捕获豆包生成图的无水印图片链接。
- Playwright 在独立线程中运行，避免阻塞 ComfyUI 的 asyncio 执行线程。

## 节点列表

节点分类均为 `Doubao/Playwright`。

| 节点名 | 输入 | 输出 | 用途 |
| --- | --- | --- | --- |
| `Doubao 文生文` | `text` | `text` | 把文本发送给豆包，返回文字回复 |
| `Doubao 文生图` | `text`、`prompt_prefix`、`image_size` | `image`、`text` | 让豆包根据文本生成图片 |
| `Doubao 图生文` | `image`、`prompt_prefix` | `text` | 上传图片，让豆包描述或分析图片 |
| `Doubao 图文生图` | `text`、`prompt_prefix`、`image_size`，可选 `image_1` 到 `image_5` | `image`、`text` | 上传参考图并让豆包生成新图 |

## 安装

1. 将本目录放到 ComfyUI 的 `custom_nodes` 目录中：

```text
ComfyUI/custom_nodes/comfyui_doubao_playwright_node
```

2. 在 ComfyUI 使用的 Python 环境中安装依赖：

```bash
pip install -r ComfyUI/custom_nodes/comfyui_doubao_playwright_node/requirements.txt
python -m playwright install chromium
```

如果你使用的是便携版 ComfyUI，需要改用便携版自带的 Python。例如：

```bash
python_embeded/python.exe -m pip install -r ComfyUI/custom_nodes/comfyui_doubao_playwright_node/requirements.txt
python_embeded/python.exe -m playwright install chromium
```

3. 重启 ComfyUI。

## 第一次使用

1. 在 ComfyUI 中添加 `Doubao/Playwright` 分类下的节点。
2. 第一次运行时，节点会打开一个可见浏览器窗口。
3. 如果豆包要求登录，请在这个浏览器窗口中手动完成登录。
4. 登录完成后重新运行节点，后续会复用同一个浏览器用户数据目录。

默认浏览器用户数据会保存在 `image_save_path/playwright_profile`。如果清空这个目录，需要重新登录。

## 参数说明

所有节点都有这些通用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `website_url` | `https://www.doubao.com/chat/` | 豆包网页地址 |
| `image_save_path` | `output/doubao_playwright` | 输入临时图和生成图保存目录；相对路径会按 ComfyUI 运行目录解析 |
| `browser_channel` | `chromium` | 可选 `chromium`、`chrome`、`msedge`；使用 `chrome` 或 `msedge` 时需要本机已安装对应浏览器 |
| `wait_timeout` | `240` | 等待回复或图片生成的最长秒数，范围 `30-1800` |
| `stable_seconds` | `4` | 页面内容保持稳定多少秒后认为回复完成，范围 `2-60` |

文生图和图文生图额外参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `prompt_prefix` | `帮我生成图片：` | 会拼接在 `text` 前面一起发给豆包 |
| `image_size` | `auto` | 可选 `auto`、`1:1`、`2:3`、`3:4`、`4:3`、`9:16`、`16:9`；非 `auto` 时会在提示词末尾追加比例描述 |

图生文额外参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `prompt_prefix` | `详细描述图片，` | 上传图片后发送给豆包的文字指令 |

## 输出行为

- `text` 输出为豆包本轮回复中提取到的文本。
- `image` 输出为本轮下载到的新图片 batch，最多 4 张。
- 如果多张输出图片尺寸不一致，节点会补边到同一尺寸后合并为 batch。
- 如果图片节点没有成功下载到图片，会返回一张 `64x64` 黑图占位，同时仍返回文本输出。
- 生成图片会保存到 `image_save_path`，上传用的临时图片会保存到 `image_save_path/_uploads`。

## 无水印扩展

节点会自动寻找并加载：

```text
browser_extension/doubao-watermark--完整版
```

如果扩展目录存在且包含 `manifest.json`，Playwright 启动浏览器时会加载它，并优先使用扩展捕获到的图片链接下载结果。若扩展不存在，节点仍会使用网页图片元素、下载按钮、右键菜单和图片资源链接等方式尝试下载。

## 使用建议

- 生图节点的 `wait_timeout` 建议保留较大值，图片生成慢时可以调到 `300` 秒以上。
- 需要固定画幅时使用 `image_size`，它本质是给豆包追加比例提示，不是 ComfyUI 本地裁剪。
- 如果豆包网页改版导致发送、上传或下载失败，先在打开的浏览器里手动确认页面能正常使用。
- 不建议最小化或关闭 Playwright 打开的浏览器窗口；关闭后节点会在下次运行时重新创建浏览器上下文。

## 常见问题

### 提示缺少 Playwright

在 ComfyUI 当前 Python 环境中重新安装依赖：

```bash
pip install -r custom_nodes/comfyui_doubao_playwright_node/requirements.txt
python -m playwright install chromium
```

### 第一次运行没有结果

通常是豆包需要登录、验证或授权。请在节点打开的浏览器中完成登录，再重新运行工作流。

### 图片节点返回黑图

黑图代表本轮没有成功下载到可用图片。常见原因包括：豆包只回复了文字、生成还没完成、网页下载入口变化、图片链接是预览小图、账号没有对应功能权限。

### 选择 `chrome` 或 `msedge` 后打不开

确认本机已安装对应浏览器。若不确定，先使用默认的 `chromium`，并确保执行过：

```bash
python -m playwright install chromium
```

## 注意事项

这个节点依赖豆包网页结构和浏览器自动化行为。网页改版、网络波动、登录状态、验证码、权限弹窗、下载按钮变化都可能影响成功率。请遵守豆包及相关网站的使用条款，不要用它绕过任何访问限制。

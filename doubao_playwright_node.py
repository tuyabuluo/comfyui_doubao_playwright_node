from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import io
import mimetypes
import re
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except Exception:  # Playwright is an optional runtime dependency for ComfyUI loading.
    PlaywrightError = RuntimeError
    PlaywrightTimeoutError = TimeoutError
    sync_playwright = None


DEFAULT_INPUT_SELECTORS = [
    'textarea[placeholder*="输入"]',
    'textarea[placeholder*="发"]',
    "textarea",
    '[contenteditable="true"][role="textbox"]',
    '[data-slate-editor="true"]',
    '[data-lexical-editor="true"]',
    ".ProseMirror",
    '[contenteditable="true"]',
    '[role="textbox"]',
]

DEFAULT_SEND_SELECTORS = [
    'button:has-text("发送")',
    'button:has-text("Send")',
    '[aria-label*="发送"]',
    '[aria-label*="Send"]',
    '[data-testid*="send"]',
    '[class*="send"]',
]

DEFAULT_NEW_CHAT_SELECTORS = [
    'button:has-text("新对话")',
    'a:has-text("新对话")',
    'button:has-text("新建对话")',
    'a:has-text("新建对话")',
    'button:has-text("New chat")',
    'a:has-text("New chat")',
    '[aria-label*="新对话"]',
    '[aria-label*="新建对话"]',
    '[aria-label*="New chat"]',
]

TEXT_ONLY_GRACE_SECONDS = 25

TEXT_CANDIDATE_SELECTORS = [
    '[data-message-author-role="assistant"]',
    '[data-testid*="assistant"]',
    '[class*="assistant"]',
    '[class*="bot"]',
    ".markdown",
    '[class*="markdown"]',
    '[data-testid*="message"]',
    '[class*="message"]',
    '[role="article"]',
    "article",
]


def _resolve_path(path_text: str, fallback: str) -> Path:
    raw = (path_text or "").strip() or fallback
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _blank_image() -> torch.Tensor:
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


def _tensor_frame_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()
    if frame.ndim == 4:
        frame = frame[0]
    arr = frame.numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    if arr.shape[-1] == 4:
        return Image.fromarray(arr, mode="RGBA")
    return Image.fromarray(arr[..., :3], mode="RGB")


def _pil_to_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _paths_to_image_batch(paths: list[Path]) -> torch.Tensor | None:
    images: list[Image.Image] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        except Exception:
            continue

    if not images:
        return None

    widths = [image.width for image in images]
    heights = [image.height for image in images]
    target_size = (max(widths), max(heights))
    if len(set(zip(widths, heights))) > 1:
        _log(f"输出图片尺寸不一致，已补边合并为 batch：{target_size[0]}x{target_size[1]}")

    tensors: list[torch.Tensor] = []
    for image in images:
        if image.size != target_size:
            canvas = Image.new("RGB", target_size)
            x = (target_size[0] - image.width) // 2
            y = (target_size[1] - image.height) // 2
            canvas.paste(image, (x, y))
            image = canvas
        arr = np.asarray(image).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr)[None, ...])

    return torch.cat(tensors, dim=0)


def _extension_from_content_type(content_type: str, fallback: str = ".png") -> str:
    content_type = (content_type or "").split(";")[0].strip().lower()
    if "png" in content_type:
        return ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        return ".jpg"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    guessed = mimetypes.guess_extension(content_type)
    return guessed or fallback


def _log(message: str) -> None:
    print(f"[Doubao AI Playwright] {message}", flush=True)


def _find_browser_extension_dir() -> Path | None:
    extension_root = Path(__file__).resolve().parent / "browser_extension"
    preferred = extension_root / "doubao-watermark--完整版"
    if (preferred / "manifest.json").is_file():
        return preferred
    if extension_root.is_dir():
        for path in extension_root.iterdir():
            if path.is_dir() and (path / "manifest.json").is_file():
                return path
    return None


class DoubaoPlaywrightSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._playwright: Any | None = None
        self._context: Any | None = None
        self._page: Any | None = None
        self._settings_key: tuple[str, str, bool, str] | None = None
        self._brought_to_front_for_context = False

    def send_and_collect(
        self,
        *,
        website_url: str,
        image_paths: list[Path],
        prompt_text: str,
        save_dir: Path,
        new_conversation: bool,
        browser_channel: str,
        browser_executable_path: str,
        user_data_dir: str,
        input_selector: str,
        upload_selector: str,
        send_button_selector: str,
        new_chat_selector: str,
        assistant_message_selector: str,
        wait_timeout: int,
        stable_seconds: int,
        min_output_images: int,
        try_hd_download: bool,
        collect_images: bool,
    ) -> tuple[list[Path], str]:
        with self._lock:
            min_output_images = max(0, min(4, int(min_output_images))) if collect_images else 0
            mode_label = "图像输出" if collect_images else "纯文本输出"
            _log(f"运行模式：{mode_label}，等待图片数：{min_output_images}")

            _log("准备打开或复用浏览器页面")
            page = self._ensure_page(
                website_url=website_url,
                save_dir=save_dir,
                browser_channel=browser_channel,
                browser_executable_path=browser_executable_path,
                user_data_dir=user_data_dir,
                headless=False,
            )
            if not self._brought_to_front_for_context:
                try:
                    page.bring_to_front()
                    _log("浏览器首次打开，已拉到前台；后续运行不会主动弹出")
                except Exception:
                    pass
                self._brought_to_front_for_context = True
            self._navigate_if_needed(page, website_url, force=False, timeout=wait_timeout)

            if new_conversation:
                _log("尝试开启新对话")
                self._open_new_chat(page, website_url, new_chat_selector, wait_timeout)

            before_extension_image_keys: set[str] = set()
            if collect_images:
                self._install_extension_image_capture(page)
                before_extension_image_keys = self._collect_extension_image_keys(page)

            if image_paths or prompt_text.strip():
                if image_paths:
                    _log(f"准备上传/粘贴 {len(image_paths)} 张图片")
                    self._upload_files(page, image_paths, upload_selector, input_selector, wait_timeout)

                if prompt_text.strip():
                    _log("准备输入文本")
                    self._insert_text(page, prompt_text, input_selector, wait_timeout)

                before_texts = self._collect_text_candidates(page, assistant_message_selector)
                before_body = self._body_text(page)
                before_image_keys: set[str] = set()
                if collect_images:
                    before_images = self._collect_image_items(page)
                    before_image_keys = {item["key"] for item in before_images}
                    before_signature = self._signature(page)
                else:
                    before_signature = self._text_signature(page)
                _log("准备发送消息")
                self._send_message(page, send_button_selector, wait_timeout)

                if collect_images:
                    _log(f"等待豆包回复稳定，至少等待 {min_output_images} 张新图片")
                    wait_result = self._wait_until_response_stable(
                        page,
                        before_signature=before_signature,
                        before_image_keys=before_image_keys,
                        before_extension_image_keys=before_extension_image_keys,
                        timeout_seconds=wait_timeout,
                        stable_seconds=stable_seconds,
                        min_output_images=min_output_images,
                    )
                else:
                    _log("等待豆包文字回复稳定")
                    wait_result = self._wait_until_text_response_stable(
                        page,
                        before_signature=before_signature,
                        assistant_message_selector=assistant_message_selector,
                        before_texts=before_texts,
                        before_body=before_body,
                        prompt_text=prompt_text,
                        timeout_seconds=wait_timeout,
                        stable_seconds=stable_seconds,
                    )
            else:
                before_texts = self._collect_text_candidates(page, assistant_message_selector)
                before_body = self._body_text(page)
                before_image_keys = set()
                wait_result = "no_input"

            generated_text = self._extract_generated_text(
                page,
                assistant_message_selector=assistant_message_selector,
                before_texts=before_texts,
                before_body=before_body,
                prompt_text=prompt_text,
            )
            _log(f"提取到文本长度 {len(generated_text)}")

            if not collect_images or wait_result == "text_only":
                _log("本轮无需输出图片，跳过图片检测和下载")
                return [], generated_text

            downloaded: list[Path] = []
            run_id = time.strftime("%Y%m%d_%H%M%S")
            extension_items = [
                item for item in self._collect_extension_image_items(page)
                if item["key"] not in before_extension_image_keys
            ]
            if extension_items:
                _log(f"插件捕获到本轮无水印图片链接 {len(extension_items)} 条，优先下载")
                downloaded.extend(
                    self._download_extension_images(
                        page,
                        extension_items[:4],
                        save_dir,
                        run_id,
                        wait_timeout,
                    )
                )
            if downloaded:
                return downloaded[:4], generated_text

            image_items = [
                item for item in self._collect_image_items(page)
                if item["key"] not in before_image_keys
            ]
            image_items = self._select_output_image_items(image_items)[:4]
            _log(f"检测到本轮新增图片 {len(image_items)} 张")
            if image_items:
                _log(
                    "准备下载候选："
                    + ", ".join(
                        f"{int(item.get('width', 0))}x{int(item.get('height', 0))}"
                        f"@({int(item.get('viewportX', item.get('x', 0)))},{int(item.get('viewportY', item.get('y', 0)))})"
                        for item in image_items
                    )
                )

            conversation_url = page.url
            for index, item in enumerate(image_items, start=1):
                stem = f"doubao_{run_id}_{index:02d}"
                saved_path: Path | None = None
                if try_hd_download:
                    saved_path = self._try_ui_original_download(page, item, save_dir, stem, conversation_url)
                if saved_path is None:
                    fallback_path = self._download_image_src(
                        page,
                        item.get("originalSrc") or item["src"],
                        save_dir,
                        stem,
                        wait_timeout,
                    )
                    if not try_hd_download or self._image_file_is_larger_than_display(fallback_path, item):
                        saved_path = fallback_path
                    elif fallback_path is not None:
                        _log("回退资源仍是预览小图，已跳过保存到输出端口")
                        try:
                            fallback_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                if saved_path is not None:
                    _log(f"已保存图片：{saved_path}")
                    downloaded.append(saved_path)
                self._restore_conversation_if_needed(page, conversation_url)

            return downloaded[:4], generated_text

    def _ensure_page(
        self,
        *,
        website_url: str,
        save_dir: Path,
        browser_channel: str,
        browser_executable_path: str,
        user_data_dir: str,
        headless: bool,
    ) -> Any:
        if sync_playwright is None:
            raise RuntimeError(
                "缺少 Playwright。请在 ComfyUI 的 Python 环境中安装："
                "pip install -r custom_nodes/comfyui_doubao_playwright_node/requirements.txt，"
                "然后执行 python -m playwright install chromium。"
            )

        profile_dir = _resolve_path(user_data_dir, str(save_dir / "playwright_profile"))
        profile_dir.mkdir(parents=True, exist_ok=True)
        channel = (browser_channel or "").strip()
        executable = (browser_executable_path or "").strip()
        extension_dir = _find_browser_extension_dir()
        extension_key = str(extension_dir) if extension_dir is not None else ""
        settings_key = (str(profile_dir), executable or channel, headless, extension_key)

        if self._page_is_alive() and self._settings_key == settings_key:
            return self._page

        self._close_context_only()
        if self._playwright is None:
            self._playwright = sync_playwright().start()

        kwargs: dict[str, Any] = {
            "user_data_dir": str(profile_dir),
            "headless": headless,
            "accept_downloads": True,
            "viewport": None,
            "no_viewport": True,
            "args": ["--start-maximized"],
        }
        if extension_dir is not None and not headless:
            extension_path = str(extension_dir)
            kwargs["args"].extend([
                f"--disable-extensions-except={extension_path}",
                f"--load-extension={extension_path}",
            ])
            _log(f"已配置加载浏览器扩展：{extension_path}")
        elif extension_dir is None:
            _log("未找到 browser_extension 下的插件目录，本次浏览器不会加载无水印扩展")

        if executable:
            kwargs["executable_path"] = executable
        elif channel and channel.lower() not in {"chromium", "bundled", "default"}:
            kwargs["channel"] = channel

        try:
            self._context = self._playwright.chromium.launch_persistent_context(**kwargs)
        except PlaywrightError:
            kwargs.pop("channel", None)
            self._context = self._playwright.chromium.launch_persistent_context(**kwargs)

        pages = [page for page in self._context.pages if not page.is_closed()]
        self._page = pages[0] if pages else self._context.new_page()
        self._page.set_default_timeout(8000)
        self._settings_key = settings_key
        return self._page

    def _page_is_alive(self) -> bool:
        try:
            return self._page is not None and not self._page.is_closed()
        except Exception:
            return False

    def _close_context_only(self) -> None:
        try:
            if self._context is not None:
                self._context.close()
        except Exception:
            pass
        self._context = None
        self._page = None
        self._brought_to_front_for_context = False

    def _navigate_if_needed(self, page: Any, website_url: str, force: bool, timeout: int) -> None:
        current = page.url or ""
        if force or current in {"", "about:blank"} or not self._same_site(current, website_url):
            page.goto(website_url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(1000)

    @staticmethod
    def _same_site(current_url: str, target_url: str) -> bool:
        try:
            current = urllib.parse.urlparse(current_url)
            target = urllib.parse.urlparse(target_url)
            return bool(current.netloc) and current.netloc == target.netloc
        except Exception:
            return False

    def _open_new_chat(self, page: Any, website_url: str, selector: str, timeout: int) -> None:
        selectors = [selector.strip()] if selector.strip() else DEFAULT_NEW_CHAT_SELECTORS
        for candidate in selectors:
            try:
                loc = page.locator(candidate).last
                if loc.count() > 0 and loc.is_visible(timeout=1500):
                    loc.click(timeout=3000)
                    page.wait_for_timeout(1500)
                    return
            except Exception:
                continue
        self._navigate_if_needed(page, website_url, force=True, timeout=timeout)

    def _upload_files(
        self,
        page: Any,
        image_paths: list[Path],
        upload_selector: str,
        input_selector: str,
        timeout: int,
    ) -> None:
        if not image_paths:
            return

        files = [str(path) for path in image_paths]
        selector = upload_selector.strip()
        if selector:
            try:
                loc = page.locator(selector).last
                tag_name = loc.evaluate("(el) => el.tagName.toLowerCase()", timeout=3000)
                input_type = loc.get_attribute("type") or ""
                if tag_name == "input" and input_type.lower() == "file":
                    loc.set_input_files(files, timeout=timeout * 1000)
                    page.wait_for_timeout(1200)
                    _log("已通过自定义 upload_selector 上传图片")
                    return
                with page.expect_file_chooser(timeout=5000) as chooser_info:
                    loc.click(timeout=3000)
                chooser_info.value.set_files(files)
                page.wait_for_timeout(1200)
                _log("已通过上传按钮选择图片")
                return
            except Exception:
                pass

        try:
            inputs = page.locator('input[type="file"]')
            count = inputs.count()
            for i in range(count - 1, -1, -1):
                try:
                    inputs.nth(i).set_input_files(files, timeout=5000)
                    page.wait_for_timeout(1200)
                    _log("已通过网页 file input 上传图片")
                    return
                except Exception:
                    continue
        except Exception:
            pass

        _log("未找到可用 file input，尝试向输入框粘贴图片")
        self._paste_files(page, image_paths, input_selector, timeout)
        _log("图片粘贴事件已发送")

    def _paste_files(self, page: Any, image_paths: list[Path], input_selector: str, timeout: int) -> None:
        textbox = self._find_textbox(page, input_selector, timeout)
        textbox.click(timeout=3000)
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            page.evaluate(
                """
                async ({ data, name, mime }) => {
                    const binary = atob(data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
                    const file = new File([bytes], name, { type: mime });
                    const transfer = new DataTransfer();
                    transfer.items.add(file);
                    const event = new ClipboardEvent("paste", {
                        clipboardData: transfer,
                        bubbles: true,
                        cancelable: true
                    });
                    const target = document.activeElement || document.body;
                    target.dispatchEvent(event);
                }
                """,
                {"data": data, "name": path.name, "mime": mime},
            )
            page.wait_for_timeout(800)

    def _insert_text(self, page: Any, prompt_text: str, input_selector: str, timeout: int) -> None:
        textbox = self._find_textbox(page, input_selector, timeout)
        textbox.click(timeout=3000)
        try:
            page.keyboard.insert_text(prompt_text)
        except Exception:
            self._dispatch_text_input(page, prompt_text)
        page.wait_for_timeout(300)
        _log("文本输入完成")

    def _find_textbox(self, page: Any, input_selector: str, timeout: int) -> Any:
        selectors = [input_selector.strip()] if input_selector.strip() else DEFAULT_INPUT_SELECTORS
        last_error: Exception | None = None
        deadline = time.monotonic() + min(max(timeout, 8), 30)
        while time.monotonic() < deadline:
            for selector in selectors:
                try:
                    loc = page.locator(selector)
                    count = loc.count()
                    for index in range(count - 1, -1, -1):
                        candidate = loc.nth(index)
                        if candidate.is_visible(timeout=300) and candidate.is_enabled(timeout=300):
                            return candidate
                except Exception as exc:
                    last_error = exc
                    continue

            try:
                element = page.evaluate_handle(
                    """
                    () => {
                        const selectors = [
                            "textarea",
                            "[contenteditable='true']",
                            "[role='textbox']",
                            "[data-slate-editor='true']",
                            "[data-lexical-editor='true']",
                            ".ProseMirror",
                            "input[type='text']",
                            "input:not([type])"
                        ];
                        const seen = new Set();
                        const candidates = [];
                        for (const selector of selectors) {
                            for (const element of document.querySelectorAll(selector)) {
                                if (seen.has(element)) continue;
                                seen.add(element);
                                const rect = element.getBoundingClientRect();
                                const style = window.getComputedStyle(element);
                                const disabled = element.disabled || element.getAttribute("aria-disabled") === "true";
                                const readonly = element.readOnly || element.getAttribute("readonly") !== null;
                                if (
                                    !disabled &&
                                    !readonly &&
                                    rect.width > 80 &&
                                    rect.height > 18 &&
                                    style.display !== "none" &&
                                    style.visibility !== "hidden" &&
                                    style.pointerEvents !== "none"
                                ) {
                                    candidates.push({ element, y: rect.y, area: rect.width * rect.height });
                                }
                            }
                        }
                        candidates.sort((a, b) => (a.y - b.y) || (a.area - b.area));
                        return candidates.length ? candidates[candidates.length - 1].element : null;
                    }
                    """
                )
                element_handle = element.as_element()
                if element_handle is not None:
                    return element_handle
            except Exception as exc:
                last_error = exc

            page.wait_for_timeout(500)

        raise RuntimeError(
            "没有找到豆包对话输入框。请确认浏览器已经登录并进入对话页，"
            "或者在节点的 input_selector 中填写当前网页输入框的选择器。"
        ) from last_error

    def _dispatch_text_input(self, page: Any, prompt_text: str) -> None:
        page.evaluate(
            """
            (text) => {
                const target = document.activeElement;
                if (!target) return;
                if (target.isContentEditable) {
                    target.textContent = `${target.textContent || ""}${text}`;
                } else if ("value" in target) {
                    target.value = `${target.value || ""}${text}`;
                }
                target.dispatchEvent(new InputEvent("input", {
                    inputType: "insertText",
                    data: text,
                    bubbles: true,
                    cancelable: true
                }));
                target.dispatchEvent(new Event("change", { bubbles: true }));
            }
            """,
            prompt_text,
        )

    def _send_message(self, page: Any, send_button_selector: str, timeout: int) -> None:
        selectors = [send_button_selector.strip()] if send_button_selector.strip() else DEFAULT_SEND_SELECTORS
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = loc.count()
                for i in range(count - 1, -1, -1):
                    button = loc.nth(i)
                    if not button.is_visible(timeout=1000):
                        continue
                    disabled = button.get_attribute("disabled")
                    aria_disabled = button.get_attribute("aria-disabled")
                    if disabled is None and aria_disabled != "true":
                        button.click(timeout=3000)
                        return
            except Exception:
                continue
        page.keyboard.press("Enter")
        page.wait_for_timeout(500)

    def _wait_until_response_stable(
        self,
        page: Any,
        *,
        before_signature: str,
        before_image_keys: set[str],
        before_extension_image_keys: set[str] | None = None,
        timeout_seconds: int,
        stable_seconds: int,
        min_output_images: int,
    ) -> str:
        deadline = time.monotonic() + timeout_seconds
        started = time.monotonic()
        last_signature = before_signature
        stable_since = time.monotonic()
        saw_change = False
        saw_generation_hint = False
        min_wait = max(2.0, min(float(stable_seconds), 4.0))
        target_images = max(0, min(4, int(min_output_images)))
        last_reported_image_count = -1
        last_reported_extension_count = -1
        before_extension_image_keys = before_extension_image_keys or set()

        while time.monotonic() < deadline:
            signature = self._signature(page)
            generating = self._looks_generating(page)
            waited = time.monotonic() - started
            new_image_count = len([
                item for item in self._collect_image_items(page)
                if item["key"] not in before_image_keys
            ])
            extension_image_count = len([
                item for item in self._collect_extension_image_items(page)
                if item["key"] not in before_extension_image_keys
            ])
            has_required_images = new_image_count >= target_images
            saw_generation_hint = saw_generation_hint or generating
            if signature != before_signature:
                saw_change = True

            if extension_image_count != last_reported_extension_count:
                last_reported_extension_count = extension_image_count
                if target_images:
                    _log(f"插件当前捕获到无水印图片 {extension_image_count}/{target_images} 张")

            if target_images and extension_image_count >= target_images and waited >= 1.0:
                if not generating or waited >= 3.0:
                    return "extension_images_ready"

            if new_image_count != last_reported_image_count:
                last_reported_image_count = new_image_count
                if target_images:
                    _log(f"当前检测到新图片 {new_image_count}/{target_images} 张")

            if signature == last_signature and not generating:
                stable_for = time.monotonic() - stable_since
                if has_required_images and saw_change and waited >= min_wait and stable_for >= stable_seconds:
                    return "images_ready" if target_images else "text_ready"
                if target_images == 0 and not saw_generation_hint and waited >= 15 and stable_for >= stable_seconds:
                    return "text_ready"
                if (
                    target_images > 0
                    and new_image_count == 0
                    and saw_change
                    and waited >= TEXT_ONLY_GRACE_SECONDS
                    and stable_for >= max(stable_seconds, 5)
                ):
                    _log(
                        "文本回复已稳定且未检测到新图片，"
                        f"{TEXT_ONLY_GRACE_SECONDS} 秒后判定为纯文字回复"
                    )
                    return "text_only"
            else:
                stable_since = time.monotonic()
                last_signature = signature

            page.wait_for_timeout(500)

        raise TimeoutError(
            f"等待豆包回复或图片生成超时（{timeout_seconds} 秒）。"
            f"当前设置要求至少检测到 {target_images} 张新图片。"
            "可以增大 wait_timeout，文本任务可把 min_output_images 设为 0，"
            "或手动检查浏览器里是否需要登录、验证或确认权限。"
        )

    def _looks_generating(self, page: Any) -> bool:
        try:
            return bool(page.evaluate(
                """
                () => {
                    const visible = (element) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        return rect.width > 1 &&
                            rect.height > 1 &&
                            style.display !== "none" &&
                            style.visibility !== "hidden";
                    };
                    const stopPattern = /停止生成|停止回答|Stop generating|Stop responding/i;
                    const statusPattern = /正在生成|生成中|思考中|Thinking|Generating/i;

                    for (const element of document.querySelectorAll("button, [role='button'], [aria-label], [title]")) {
                        if (!visible(element)) continue;
                        const label = [
                            element.innerText || "",
                            element.getAttribute("aria-label") || "",
                            element.getAttribute("title") || ""
                        ].join(" ");
                        if (stopPattern.test(label)) return true;
                    }

                    for (const element of document.querySelectorAll("[aria-live], [class*='loading'], [class*='generat'], [class*='thinking']")) {
                        if (!visible(element)) continue;
                        const text = (element.innerText || element.textContent || "").trim();
                        if (text.length <= 80 && statusPattern.test(text)) return true;
                    }
                    return false;
                }
                """
            ))
        except Exception:
            return False

    def _wait_until_text_response_stable(
        self,
        page: Any,
        *,
        before_signature: str,
        assistant_message_selector: str,
        before_texts: list[str],
        before_body: str,
        prompt_text: str,
        timeout_seconds: int,
        stable_seconds: int,
    ) -> str:
        del before_signature
        deadline = time.monotonic() + timeout_seconds
        started = time.monotonic()
        last_text = ""
        text_stable_since = time.monotonic()
        saw_change = False
        min_wait = 1.5

        while time.monotonic() < deadline:
            generating = self._looks_generating(page)
            current_text = self._extract_generated_text(
                page,
                assistant_message_selector=assistant_message_selector,
                before_texts=before_texts,
                before_body=before_body,
                prompt_text=prompt_text,
            )
            if current_text:
                saw_change = True

            if current_text != last_text:
                last_text = current_text
                text_stable_since = time.monotonic()

            if current_text and not generating:
                stable_for = time.monotonic() - text_stable_since
                waited = time.monotonic() - started
                required_stable_seconds = self._required_text_stable_seconds(
                    current_text,
                    stable_seconds,
                    prompt_text,
                )
                if saw_change and waited >= min_wait and stable_for >= required_stable_seconds:
                    return "text_ready"

            page.wait_for_timeout(300)

        raise TimeoutError(
            f"等待豆包文字回复超时（{timeout_seconds} 秒）。"
            "可以增大 wait_timeout，或手动检查浏览器里是否需要登录、验证或确认权限。"
        )

    @staticmethod
    def _required_text_stable_seconds(text: str, stable_seconds: int, prompt_text: str) -> float:
        text_length = len(_clean_text(text))
        configured = float(stable_seconds)
        prompt_clean = _clean_text(prompt_text)
        expects_long_answer = bool(re.search(r"详细|描述|分析|说明|describe|detail|analyze", prompt_clean, re.I))
        if expects_long_answer and text_length < 60:
            return max(5.0, min(configured, 8.0))
        if expects_long_answer and text_length < 120:
            return max(2.5, min(configured, 5.0))
        if text_length < 12:
            return max(4.0, min(configured, 6.0))
        if text_length < 50:
            return max(2.5, min(configured, 4.0))
        return max(1.2, min(configured, 1.8))

    def _text_signature(self, page: Any) -> str:
        try:
            return page.evaluate(
                """
                () => (document.body?.innerText || "").slice(-10000)
                """
            )
        except Exception:
            return str(time.time())

    def _signature(self, page: Any) -> str:
        try:
            return page.evaluate(
                """
                () => {
                    const text = (document.body?.innerText || "").slice(-8000);
                    const images = Array.from(document.images)
                        .map((img) => [
                            img.currentSrc || img.src || "",
                            img.naturalWidth || 0,
                            img.naturalHeight || 0
                        ].join(":"))
                        .join("|");
                    return `${text}\\n__IMAGES__${images}`;
                }
                """
            )
        except Exception:
            return str(time.time())

    def _body_text(self, page: Any) -> str:
        try:
            return _clean_text(page.locator("body").inner_text(timeout=2000))
        except Exception:
            return ""

    def _install_extension_image_capture(self, page: Any) -> None:
        try:
            page.evaluate(
                """
                () => {
                    if (window.__doubaoNodeImageCaptureInstalled) return;
                    window.__doubaoNodeImageCaptureInstalled = true;
                    window.__doubaoNodeImageData = window.__doubaoNodeImageData || [];
                    window.addEventListener("message", (event) => {
                        const payload = event?.data || {};
                        if (
                            payload.type !== "imageDataExtracted" &&
                            payload.type !== "aptpreset_doubao_image_data"
                        ) {
                            return;
                        }
                        const items = Array.isArray(payload.data) ? payload.data : [];
                        for (const item of items) {
                            const url = item?.no_watermark_url || item?.watermark_url;
                            if (!url) continue;
                            window.__doubaoNodeImageData.push({
                                no_watermark_url: item.no_watermark_url || "",
                                watermark_url: item.watermark_url || "",
                                width: item.width || 0,
                                height: item.height || 0,
                                captured_at: Date.now()
                            });
                        }
                    }, true);
                }
                """
            )
        except Exception:
            pass

    def _collect_extension_image_items(self, page: Any) -> list[dict[str, Any]]:
        try:
            items = page.evaluate(
                """
                () => {
                    const seen = new Set();
                    const output = [];
                    for (const item of window.__doubaoNodeImageData || []) {
                        const url = item.no_watermark_url || item.watermark_url || "";
                        if (!url || seen.has(url)) continue;
                        seen.add(url);
                        output.push({
                            key: url,
                            noWatermarkUrl: item.no_watermark_url || "",
                            watermarkUrl: item.watermark_url || "",
                            width: item.width || 0,
                            height: item.height || 0,
                            capturedAt: item.captured_at || 0
                        });
                    }
                    return output;
                }
                """
            )
            return list(items)
        except Exception:
            return []

    def _collect_extension_image_keys(self, page: Any) -> set[str]:
        return {item["key"] for item in self._collect_extension_image_items(page)}

    def _download_extension_images(
        self,
        page: Any,
        items: list[dict[str, Any]],
        save_dir: Path,
        run_id: str,
        timeout: int,
    ) -> list[Path]:
        downloaded: list[Path] = []
        for index, item in enumerate(items, start=1):
            url = item.get("noWatermarkUrl") or item.get("watermarkUrl") or ""
            if not url:
                continue
            stem = f"doubao_{run_id}_{index:02d}_no_watermark"
            path = self._download_image_src(page, url, save_dir, stem, timeout)
            if path is not None:
                _log(f"已通过插件无水印链接保存图片：{path}")
                downloaded.append(path)
        return downloaded

    def _collect_text_candidates(self, page: Any, selector: str) -> list[str]:
        selectors = [selector.strip()] if selector.strip() else TEXT_CANDIDATE_SELECTORS
        texts: list[str] = []
        for candidate in selectors:
            try:
                for text in page.locator(candidate).all_inner_texts():
                    cleaned = _clean_text(text)
                    if cleaned and cleaned not in texts:
                        texts.append(cleaned)
                if selector.strip() and texts:
                    break
            except Exception:
                continue
        for text in self._collect_parent_text_candidates(page, selector):
            if text and text not in texts:
                texts.append(text)
        return texts

    def _collect_parent_text_candidates(self, page: Any, selector: str) -> list[str]:
        selectors = [selector.strip()] if selector.strip() else TEXT_CANDIDATE_SELECTORS
        try:
            texts = page.evaluate(
                """
                (selectors) => {
                    const blockedTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "SVG", "BUTTON", "INPUT", "TEXTAREA"]);
                    const clean = (text) => (text || "").replace(/\\r\\n/g, "\\n").replace(/\\r/g, "\\n").replace(/[ \\t]+\\n/g, "\\n").replace(/\\n{3,}/g, "\\n\\n").trim();
                    const visible = (element) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        return rect.width > 1 &&
                            rect.height > 1 &&
                            style.display !== "none" &&
                            style.visibility !== "hidden";
                    };
                    const looksLikeMessage = (element) => {
                        const attrs = [
                            element.getAttribute("class") || "",
                            element.getAttribute("role") || "",
                            element.getAttribute("data-testid") || "",
                            element.getAttribute("data-message-author-role") || ""
                        ].join(" ");
                        return /assistant|bot|markdown|message|chat|answer|article|content|bubble|item/i.test(attrs);
                    };
                    const output = [];
                    const seen = new Set();
                    const push = (element) => {
                        if (!element || blockedTags.has(element.tagName) || !visible(element)) return;
                        const text = clean(element.innerText || element.textContent || "");
                        if (text.length < 2 || seen.has(text)) return;
                        seen.add(text);
                        output.push(text);
                    };

                    for (const selector of selectors) {
                        for (const element of document.querySelectorAll(selector)) {
                            const baseText = clean(element.innerText || element.textContent || "");
                            if (!baseText) continue;
                            push(element);
                            let current = element.parentElement;
                            for (let depth = 0; depth < 6 && current && current !== document.body; depth += 1) {
                                const currentText = clean(current.innerText || current.textContent || "");
                                if (
                                    currentText.includes(baseText) &&
                                    currentText.length <= Math.max(baseText.length + 6000, baseText.length * 8) &&
                                    (depth <= 2 || looksLikeMessage(current))
                                ) {
                                    push(current);
                                }
                                current = current.parentElement;
                            }
                        }
                    }
                    return output;
                }
                """,
                selectors,
            )
            return [_clean_text(str(text)) for text in texts if _clean_text(str(text))]
        except Exception:
            return []

    def _extract_generated_text(
        self,
        page: Any,
        *,
        assistant_message_selector: str,
        before_texts: list[str],
        before_body: str,
        prompt_text: str,
    ) -> str:
        after_texts = self._collect_text_candidates(page, assistant_message_selector)
        before_set = set(before_texts)
        before_blocks = set(self._split_text_blocks("\n".join(before_texts) + "\n" + before_body))
        prompt_clean = _clean_text(prompt_text)
        new_texts = [
            self._current_reply_from_text(text, before_blocks, prompt_clean)
            for text in after_texts
            if text not in before_set
        ]
        new_texts = [text for text in new_texts if text]
        if new_texts:
            return self._best_reply_candidate(new_texts)

        after_blocks = self._collect_visible_text_blocks(page)
        new_blocks = [
            text for text in after_blocks
            if text not in before_blocks and not self._is_user_or_ui_text(text, prompt_clean)
        ]
        if new_blocks:
            return _clean_text("\n".join(new_blocks))

        after_body = self._body_text(page)
        suffix = self._body_text_delta(before_body, after_body)
        if suffix:
            suffix_blocks = [
                text for text in self._split_text_blocks(suffix)
                if not self._is_user_or_ui_text(text, prompt_clean)
            ]
            if suffix_blocks:
                return _clean_text("\n".join(suffix_blocks))
        return ""

    @staticmethod
    def _best_reply_candidate(texts: list[str]) -> str:
        unique_texts = list(dict.fromkeys(_clean_text(text) for text in texts if _clean_text(text)))
        if not unique_texts:
            return ""
        return max(
            unique_texts,
            key=lambda text: (
                len(DoubaoPlaywrightSession._split_text_blocks(text)),
                len(text),
            ),
        )

    @staticmethod
    def _current_reply_from_text(text: str, before_blocks: set[str], prompt_text: str) -> str:
        cleaned = _clean_text(text)
        if not cleaned or DoubaoPlaywrightSession._is_user_or_ui_text(cleaned, prompt_text):
            return ""

        blocks = DoubaoPlaywrightSession._split_text_blocks(cleaned)
        fresh_blocks = [
            block for block in blocks
            if block not in before_blocks and not DoubaoPlaywrightSession._is_user_or_ui_text(block, prompt_text)
        ]
        if fresh_blocks and len(fresh_blocks) < len(blocks):
            return _clean_text("\n".join(fresh_blocks))
        if fresh_blocks:
            return cleaned
        return ""

    def _collect_visible_text_blocks(self, page: Any) -> list[str]:
        try:
            blocks = page.evaluate(
                """
                () => {
                    const blockedTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "SVG", "BUTTON", "INPUT", "TEXTAREA"]);
                    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    const items = [];
                    const seen = new Set();
                    let node;
                    while ((node = walker.nextNode())) {
                        const text = (node.nodeValue || "").replace(/\\s+/g, " ").trim();
                        if (!text || text.length < 2 || seen.has(text)) continue;
                        const parent = node.parentElement;
                        if (!parent || blockedTags.has(parent.tagName)) continue;
                        const rect = parent.getBoundingClientRect();
                        const style = window.getComputedStyle(parent);
                        if (
                            rect.width <= 1 ||
                            rect.height <= 1 ||
                            style.display === "none" ||
                            style.visibility === "hidden"
                        ) continue;
                        seen.add(text);
                        items.push({ text, y: rect.y + window.scrollY, x: rect.x + window.scrollX });
                    }
                    items.sort((a, b) => (a.y - b.y) || (a.x - b.x));
                    return items.map((item) => item.text);
                }
                """
            )
            return [_clean_text(str(text)) for text in blocks if _clean_text(str(text))]
        except Exception:
            return []

    @staticmethod
    def _split_text_blocks(text: str) -> list[str]:
        return [
            _clean_text(line)
            for line in re.split(r"\n+|\s{2,}", text or "")
            if _clean_text(line)
        ]

    @staticmethod
    def _body_text_delta(before_body: str, after_body: str) -> str:
        if not after_body:
            return ""
        if before_body and after_body.startswith(before_body):
            return _clean_text(after_body[len(before_body):])

        before_lines = set(DoubaoPlaywrightSession._split_text_blocks(before_body))
        after_lines = DoubaoPlaywrightSession._split_text_blocks(after_body)
        delta = [line for line in after_lines if line not in before_lines]
        return _clean_text("\n".join(delta))

    @staticmethod
    def _is_user_or_ui_text(text: str, prompt_text: str) -> bool:
        cleaned = _clean_text(text)
        if not cleaned:
            return True
        if prompt_text and cleaned == prompt_text:
            return True
        if prompt_text and cleaned in prompt_text:
            return True
        ui_texts = {
            "发送",
            "重新生成",
            "复制",
            "分享",
            "下载",
            "下载高清",
            "高清",
            "原图",
            "停止生成",
            "停止回答",
            "正在生成",
            "生成中",
            "思考中",
            "Send",
            "Copy",
            "Share",
            "Download",
            "Stop generating",
            "Generating",
        }
        return cleaned in ui_texts

    def _collect_image_items(self, page: Any) -> list[dict[str, Any]]:
        try:
            items = page.evaluate(
                """
                () => {
                    const items = [];
                    let index = 0;
                    const normalize = (url) => {
                        if (!url) return "";
                        try {
                            return new URL(url, document.baseURI).href;
                        } catch {
                            return url;
                        }
                    };
                    const addCandidate = (candidates, url, priority, score) => {
                        const normalized = normalize(url);
                        if (!normalized || normalized.startsWith("data:image/svg")) return;
                        candidates.push({ url: normalized, priority, score: score || 0 });
                    };
                    const srcsetCandidates = (srcset) => {
                        if (!srcset) return [];
                        return srcset.split(",").map((part) => {
                            const bits = part.trim().split(/\\s+/);
                            const url = bits[0] || "";
                            const descriptor = bits[1] || "";
                            let score = 0;
                            if (descriptor.endsWith("w")) score = Number.parseInt(descriptor, 10) || 0;
                            if (descriptor.endsWith("x")) score = (Number.parseFloat(descriptor) || 0) * 1000;
                            return { url, score };
                        }).filter((item) => item.url);
                    };
                    const candidateSrc = (element, fallbackSrc, naturalWidth, naturalHeight) => {
                        const candidates = [];
                        addCandidate(candidates, fallbackSrc, 10, naturalWidth * naturalHeight);

                        if (element instanceof HTMLImageElement) {
                            addCandidate(candidates, element.src, 12, naturalWidth * naturalHeight);
                            addCandidate(candidates, element.currentSrc, 14, naturalWidth * naturalHeight);
                            for (const item of srcsetCandidates(element.srcset)) {
                                addCandidate(candidates, item.url, 80, item.score);
                            }
                        }

                        let current = element;
                        let depth = 0;
                        while (current && depth < 6) {
                            if (current.dataset) {
                                for (const [key, value] of Object.entries(current.dataset)) {
                                    if (/origin|original|raw|source|large|hd|high|download|url|src/i.test(key)) {
                                        addCandidate(current === element ? candidates : candidates, value, 70 - depth, 0);
                                    }
                                }
                            }
                            const href = current.getAttribute?.("href");
                            if (href && /image|img|origin|original|large|hd|download|tos|obj|byte/i.test(href)) {
                                addCandidate(candidates, href, 75 - depth, 0);
                            }
                            current = current.parentElement;
                            depth += 1;
                        }

                        candidates.sort((a, b) => (b.priority - a.priority) || (b.score - a.score) || (b.url.length - a.url.length));
                        return candidates.length ? candidates[0].url : normalize(fallbackSrc);
                    };
                    const pushItem = (element, src, naturalWidth, naturalHeight, kind) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        const width = naturalWidth || Math.round(rect.width);
                        const height = naturalHeight || Math.round(rect.height);
                        const normalizedSrc = normalize(src);
                        const originalSrc = candidateSrc(element, normalizedSrc, width, height);
                        const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1;
                        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 1;
                        const visible = src &&
                            style.visibility !== "hidden" &&
                            style.display !== "none" &&
                            rect.width > 32 &&
                            rect.height > 32;
                        if (
                            visible &&
                            width >= 96 &&
                            height >= 96 &&
                            !src.startsWith("data:image/svg")
                        ) {
                            items.push({
                                src: normalizedSrc,
                                originalSrc,
                                key: `${kind}:${normalizedSrc}|${width}|${height}`,
                                index: index++,
                                width,
                                height,
                                area: width * height,
                                x: rect.x + window.scrollX,
                                y: rect.y + window.scrollY,
                                viewportX: rect.x,
                                viewportY: rect.y,
                                viewportWidth,
                                viewportHeight,
                                inMainArea: rect.x > Math.min(260, viewportWidth * 0.24) && rect.x < viewportWidth * 0.86,
                                nearBottom: rect.y > viewportHeight * 0.20,
                                visible,
                                kind
                            });
                        }
                    };

                    for (const img of Array.from(document.images)) {
                        pushItem(img, img.currentSrc || img.src || "", img.naturalWidth, img.naturalHeight, "img");
                    }

                    for (const element of Array.from(document.querySelectorAll("*"))) {
                        const background = window.getComputedStyle(element).backgroundImage;
                        if (!background || background === "none") continue;
                        const matches = Array.from(background.matchAll(/url\\(["']?([^"')]+)["']?\\)/g));
                        for (const match of matches) {
                            pushItem(element, match[1], 0, 0, "background");
                        }
                    }
                    return items;
                }
                """
            )
            return list(items)
        except Exception:
            return []

    @staticmethod
    def _dedupe_images(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_src: dict[str, dict[str, Any]] = {}
        for item in items:
            src = item["src"]
            previous = by_src.get(src)
            if previous is None or item.get("area", 0) >= previous.get("area", 0):
                by_src[src] = item
        return sorted(by_src.values(), key=lambda item: item.get("index", 0))

    def _select_output_image_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped = self._dedupe_images(items)
        main_items = [item for item in deduped if item.get("inMainArea")]
        candidates = main_items or deduped
        candidates = [
            item for item in candidates
            if item.get("area", 0) >= 180 * 120 and item.get("nearBottom", True)
        ] or candidates
        candidates = sorted(
            candidates,
            key=lambda item: (
                float(item.get("y", 0)),
                1 if item.get("inMainArea") else 0,
                float(item.get("area", 0)),
                float(item.get("x", 0)),
            ),
            reverse=True,
        )
        selected = sorted(candidates[:4], key=lambda item: (float(item.get("y", 0)), float(item.get("x", 0))))
        return selected

    def _try_ui_original_download(
        self,
        page: Any,
        item: dict[str, Any],
        save_dir: Path,
        stem: str,
        conversation_url: str,
    ) -> Path | None:
        src = item.get("src") or ""
        original_src = item.get("originalSrc") or src
        try:
            hover_path = self._try_hover_card_download(page, item, save_dir, stem)
            if hover_path is not None:
                self._log_image_file_info(hover_path, "已通过图片下方下载按钮保存")
                return hover_path

            clicked = self._click_image_item(page, item)
            if not clicked:
                return None
            page.wait_for_timeout(1200)

            original_download_selectors = [
                'button:has-text("下载原图")',
                'a:has-text("下载原图")',
                'button:has-text("保存原图")',
                'a:has-text("保存原图")',
                'button:has-text("查看原图")',
                'a:has-text("查看原图")',
                'button:has-text("打开原图")',
                'a:has-text("打开原图")',
                'button:has-text("原始图片")',
                'a:has-text("原始图片")',
                'button:has-text("原图下载")',
                'a:has-text("原图下载")',
                'button:has-text("高清原图")',
                'a:has-text("高清原图")',
                'button:has-text("下载高清")',
                'a:has-text("下载高清")',
                'button:has-text("高清下载")',
                'a:has-text("高清下载")',
                'button:has-text("原图")',
                'a:has-text("原图")',
                '[aria-label*="下载原图"]',
                '[title*="下载原图"]',
                '[aria-label*="原图"]',
                '[title*="原图"]',
                'button:has-text("无水印")',
                'a:has-text("无水印")',
            ]
            fallback_download_selectors = [
                'button:has-text("保存")',
                'a:has-text("保存")',
                'button:has-text("保存图片")',
                'a:has-text("保存图片")',
                'button:has-text("下载图片")',
                'a:has-text("下载图片")',
                '[aria-label*="保存"]',
                '[title*="保存"]',
                'button:has-text("高清")',
                'a:has-text("高清")',
                'button:has-text("下载")',
                'a:has-text("下载")',
                'a[download]:has-text("原图")',
                'a[download]:has-text("高清")',
                'a[download]',
                '[aria-label*="下载"]',
                '[title*="下载"]',
            ]

            path = self._try_viewer_save_button(page, fallback_download_selectors, save_dir, stem)
            if path is not None:
                self._log_image_file_info(path, "已通过右侧大图保存按钮保存")
                return path

            if self._right_click_viewer_image(page):
                _log("已在右侧大图上打开右键菜单，查找下载原图")
                path = self._try_download_controls(
                    page,
                    original_download_selectors + fallback_download_selectors,
                    save_dir,
                    stem,
                    prefer_right_side=True,
                )
                if path is not None:
                    self._log_image_file_info(path, "已通过右键菜单下载原图")
                    return path
                try:
                    page.keyboard.press("Escape")
                except Exception:
                    pass

            path = self._try_download_controls(
                page,
                original_download_selectors + fallback_download_selectors,
                save_dir,
                stem,
                prefer_right_side=True,
            )
            if path is not None:
                self._log_image_file_info(path, "已通过右侧大图下载按钮保存")
                return path

            if original_src and original_src != src:
                path = self._download_image_src(page, original_src, save_dir, f"{stem}_original", 60)
                if path is not None:
                    return path
        except Exception:
            return None
        finally:
            self._restore_conversation_if_needed(page, conversation_url)
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
        return None

    def _try_hover_card_download(
        self,
        page: Any,
        item: dict[str, Any],
        save_dir: Path,
        stem: str,
    ) -> Path | None:
        rect = self._get_image_item_rect(page, item)
        if not rect:
            return None
        try:
            x = float(rect["x"]) + float(rect["width"]) / 2
            y = float(rect["y"]) + max(8.0, float(rect["height"]) - 10.0)
            page.mouse.move(x, y)
            page.wait_for_timeout(900)
            _log("已悬停生成图，查找图片下方下载按钮")
            selectors = [
                'button:has-text("下载")',
                'a:has-text("下载")',
                'div[role="button"]:has-text("下载")',
                '[role="button"]:has-text("下载")',
                '[aria-label*="下载"]',
                '[title*="下载"]',
                '[data-testid*="download"]',
            ]
            return self._try_download_controls_near_rect(page, selectors, save_dir, stem, rect)
        except Exception:
            return None

    def _get_image_item_rect(self, page: Any, item: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return page.evaluate(
                """
                (item) => {
                    const normalize = (url) => {
                        if (!url) return "";
                        try {
                            return new URL(url, document.baseURI).href;
                        } catch {
                            return url;
                        }
                    };
                    const src = normalize(item.src || "");
                    const originalSrc = normalize(item.originalSrc || src);
                    const expectedX = Number(item.x || 0);
                    const expectedY = Number(item.y || 0);
                    const candidates = [];
                    const push = (element) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        if (
                            style.display === "none" ||
                            style.visibility === "hidden" ||
                            rect.width < 64 ||
                            rect.height < 64
                        ) return;
                        const docX = rect.x + window.scrollX;
                        const docY = rect.y + window.scrollY;
                        const distance = Math.abs(docX - expectedX) + Math.abs(docY - expectedY);
                        candidates.push({ element, area: rect.width * rect.height, distance });
                    };
                    for (const img of Array.from(document.images)) {
                        const imgSrc = normalize(img.currentSrc || img.src || "");
                        const imgOriginal = normalize(img.src || "");
                        if (imgSrc === src || imgOriginal === src || imgSrc === originalSrc || imgOriginal === originalSrc) {
                            push(img);
                        }
                    }
                    for (const element of Array.from(document.querySelectorAll("*"))) {
                        const background = window.getComputedStyle(element).backgroundImage || "";
                        if (background.includes(src) || background.includes(originalSrc)) {
                            push(element);
                        }
                    }
                    candidates.sort((a, b) => (a.distance - b.distance) || (b.area - a.area));
                    const target = candidates[0]?.element;
                    if (!target) return null;
                    target.scrollIntoView({ block: "center", inline: "center" });
                    const rect = target.getBoundingClientRect();
                    return {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        viewportWidth: window.innerWidth || document.documentElement.clientWidth || 1,
                        viewportHeight: window.innerHeight || document.documentElement.clientHeight || 1
                    };
                }
                """,
                item,
            )
        except Exception:
            return None

    def _click_image_item(self, page: Any, item: dict[str, Any]) -> bool:
        rect = self._get_image_item_rect(page, item)
        if not rect:
            return False
        try:
            page.mouse.click(
                float(rect["x"]) + float(rect["width"]) / 2,
                float(rect["y"]) + float(rect["height"]) / 2,
            )
            return True
        except Exception:
            return False

    def _try_viewer_save_button(
        self,
        page: Any,
        selectors: list[str],
        save_dir: Path,
        stem: str,
    ) -> Path | None:
        page.wait_for_timeout(1000)
        _log("已打开右侧大图，查找右上角保存按钮")
        return self._try_download_controls(
            page,
            selectors,
            save_dir,
            stem,
            prefer_right_side=True,
        ) or self._try_top_right_icon_download(page, save_dir, stem)

    def _right_click_viewer_image(self, page: Any) -> bool:
        try:
            rect = page.evaluate(
                """
                () => {
                    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1;
                    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 1;
                    const items = [];
                    const push = (element) => {
                        const rect = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        if (
                            style.display === "none" ||
                            style.visibility === "hidden" ||
                            rect.width < 120 ||
                            rect.height < 120 ||
                            rect.x < viewportWidth * 0.38
                        ) return;
                        items.push({
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            area: rect.width * rect.height
                        });
                    };
                    for (const img of Array.from(document.images)) push(img);
                    for (const element of Array.from(document.querySelectorAll("*"))) {
                        const background = window.getComputedStyle(element).backgroundImage;
                        if (background && background !== "none") push(element);
                    }
                    items.sort((a, b) => b.area - a.area);
                    const item = items[0];
                    if (!item) return null;
                    return {
                        x: item.x + item.width / 2,
                        y: item.y + item.height / 2,
                        width: item.width,
                        height: item.height
                    };
                }
                """
            )
            if not rect:
                return False
            page.mouse.click(float(rect["x"]), float(rect["y"]), button="right")
            page.wait_for_timeout(600)
            return True
        except Exception:
            return False

    def _try_download_controls(
        self,
        page: Any,
        selectors: list[str],
        save_dir: Path,
        stem: str,
        *,
        prefer_right_side: bool,
    ) -> Path | None:
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = loc.count()
                for index in range(count - 1, -1, -1):
                    candidate = loc.nth(index)
                    if not candidate.is_visible(timeout=700):
                        continue
                    if prefer_right_side and not self._is_right_side_control(candidate):
                        continue
                    path = self._save_from_download_control(page, candidate, save_dir, stem)
                    if path is not None:
                        return path
            except Exception:
                continue
        return None

    def _try_download_controls_near_rect(
        self,
        page: Any,
        selectors: list[str],
        save_dir: Path,
        stem: str,
        rect: dict[str, Any],
    ) -> Path | None:
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = loc.count()
                for index in range(count - 1, -1, -1):
                    candidate = loc.nth(index)
                    if not candidate.is_visible(timeout=700):
                        continue
                    if not self._is_control_near_rect(candidate, rect):
                        continue
                    path = self._save_from_download_control(page, candidate, save_dir, stem)
                    if path is not None:
                        return path
            except Exception:
                continue
        return None

    def _try_top_right_icon_download(self, page: Any, save_dir: Path, stem: str) -> Path | None:
        selectors = [
            '[data-testid*="save"]',
            '[data-testid*="download"]',
            '[class*="save"]',
            '[class*="Save"]',
            '[class*="download"]',
            '[class*="Download"]',
        ]
        for selector in selectors:
            try:
                loc = page.locator(selector)
                count = loc.count()
                for index in range(count - 1, -1, -1):
                    candidate = loc.nth(index)
                    if not candidate.is_visible(timeout=500):
                        continue
                    if not self._is_top_right_viewer_control(candidate):
                        continue
                    _log("尝试点击右侧大图右上角保存图标")
                    path = self._save_from_download_control(page, candidate, save_dir, stem)
                    if path is not None:
                        return path
            except Exception:
                continue
        return None

    @staticmethod
    def _is_top_right_viewer_control(loc: Any) -> bool:
        try:
            rect = loc.evaluate(
                """
                (element) => {
                    const rect = element.getBoundingClientRect();
                    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1;
                    const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 1;
                    const label = [
                        element.getAttribute("aria-label") || "",
                        element.getAttribute("title") || "",
                        element.textContent || ""
                    ].join(" ");
                    return {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        viewportWidth,
                        viewportHeight,
                        label
                    };
                }
                """
            )
            label = str(rect.get("label") or "")
            if re.search(r"关闭|取消|返回|close|cancel|back", label, re.I):
                return False
            if rect["width"] <= 1 or rect["height"] <= 1:
                return False
            center_x = rect["x"] + rect["width"] / 2
            center_y = rect["y"] + rect["height"] / 2
            return center_x >= rect["viewportWidth"] * 0.55 and center_y <= rect["viewportHeight"] * 0.25
        except Exception:
            return False

    @staticmethod
    def _is_control_near_rect(loc: Any, target_rect: dict[str, Any]) -> bool:
        try:
            rect = loc.evaluate(
                """
                (element) => {
                    const rect = element.getBoundingClientRect();
                    return {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    };
                }
                """
            )
            if rect["width"] <= 1 or rect["height"] <= 1:
                return False
            control_x = float(rect["x"]) + float(rect["width"]) / 2
            control_y = float(rect["y"]) + float(rect["height"]) / 2
            image_x = float(target_rect["x"])
            image_y = float(target_rect["y"])
            image_w = float(target_rect["width"])
            image_h = float(target_rect["height"])
            return (
                image_x - 80 <= control_x <= image_x + image_w + 80
                and image_y - 60 <= control_y <= image_y + image_h + 130
            )
        except Exception:
            return False

    @staticmethod
    def _is_right_side_control(loc: Any) -> bool:
        try:
            rect = loc.evaluate(
                """
                (element) => {
                    const rect = element.getBoundingClientRect();
                    const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 1;
                    return {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        viewportWidth
                    };
                }
                """
            )
            if rect["width"] <= 1 or rect["height"] <= 1:
                return False
            return rect["x"] >= rect["viewportWidth"] * 0.35
        except Exception:
            return False

    def _save_from_download_control(self, page: Any, loc: Any, save_dir: Path, stem: str) -> Path | None:
        href = ""
        try:
            href = loc.get_attribute("href") or ""
        except Exception:
            href = ""

        if href and not href.lower().startswith(("javascript:", "#")):
            direct_path = self._download_image_src(page, href, save_dir, f"{stem}_original", 120)
            if direct_path is not None:
                return direct_path

        try:
            _log("已点击下载/保存按钮，等待豆包准备原图下载")
            with page.expect_download(timeout=120000) as download_info:
                loc.click(timeout=5000)
            download = download_info.value
            suffix = Path(download.suggested_filename or "").suffix or ".png"
            path = save_dir / f"{stem}_original{suffix}"
            download.save_as(str(path))
            if self._is_valid_image_file(path):
                return path
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception:
            return None
        return None

    def _restore_conversation_if_needed(self, page: Any, conversation_url: str) -> None:
        try:
            current_url = page.url
            if conversation_url and current_url and current_url != conversation_url:
                _log("检测到下载操作切换了页面，正在恢复当前对话")
                page.goto(conversation_url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(1000)
        except Exception:
            pass

    @staticmethod
    def _is_valid_image_file(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                image.verify()
            return True
        except Exception:
            return False

    @staticmethod
    def _log_image_file_info(path: Path, prefix: str) -> None:
        try:
            with Image.open(path) as image:
                width, height = image.size
            _log(f"{prefix}：{path} ({width}x{height})")
        except Exception:
            _log(f"{prefix}：{path}")

    @staticmethod
    def _image_file_is_larger_than_display(path: Path | None, item: dict[str, Any]) -> bool:
        if path is None:
            return False
        try:
            with Image.open(path) as image:
                width, height = image.size
            display_width = max(1, int(item.get("width") or 1))
            display_height = max(1, int(item.get("height") or 1))
            file_area = width * height
            display_area = display_width * display_height
            return width >= 1024 or height >= 1024 or file_area >= display_area * 2
        except Exception:
            return False

    def _download_image_src(self, page: Any, src: str, save_dir: Path, stem: str, timeout: int) -> Path | None:
        try:
            src = urllib.parse.urljoin(page.url, src)
            if src.startswith("data:"):
                header, data = src.split(",", 1)
                content_type = header[5:].split(";")[0]
                raw = base64.b64decode(data) if ";base64" in header else urllib.parse.unquote_to_bytes(data)
                ext = _extension_from_content_type(content_type)
            elif src.startswith("blob:"):
                payload = page.evaluate(
                    """
                    async (src) => {
                        const response = await fetch(src);
                        const buffer = await response.arrayBuffer();
                        const bytes = new Uint8Array(buffer);
                        let binary = "";
                        const chunk = 0x8000;
                        for (let i = 0; i < bytes.length; i += chunk) {
                            binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
                        }
                        return {
                            contentType: response.headers.get("content-type") || "image/png",
                            data: btoa(binary)
                        };
                    }
                    """,
                    src,
                )
                raw = base64.b64decode(payload["data"])
                ext = _extension_from_content_type(payload.get("contentType", "image/png"))
            else:
                response = page.context.request.get(src, timeout=timeout * 1000)
                if not response.ok:
                    return None
                raw = response.body()
                ext = _extension_from_content_type(response.headers.get("content-type", "image/png"))

            try:
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    image_format = (image.format or "").lower()
                    if image_format == "jpeg":
                        ext = ".jpg"
                    elif image_format in {"png", "webp", "gif", "bmp", "tiff"}:
                        ext = f".{image_format}"
            except Exception:
                _log("下载链接返回的不是有效图片，已跳过")
                return None

            path = save_dir / f"{stem}{ext}"
            path.write_bytes(raw)
            _log(f"直接保存图片资源：{width}x{height}")
            return path
        except Exception:
            return None


class DoubaoPlaywrightWorker:
    """Run Playwright's sync API outside ComfyUI's asyncio execution thread."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="doubao-playwright",
        )
        self._session: DoubaoPlaywrightSession | None = None

    def send_and_collect(self, **kwargs: Any) -> tuple[list[Path], str]:
        future = self._executor.submit(self._send_and_collect, kwargs)
        return future.result()

    def _send_and_collect(self, kwargs: dict[str, Any]) -> tuple[list[Path], str]:
        if self._session is None:
            self._session = DoubaoPlaywrightSession()
        return self._session.send_and_collect(**kwargs)


SESSION = DoubaoPlaywrightWorker()


RATIO_SUFFIXES = {
    "auto": "",
    "1:1": "，比例 「1:1」",
    "2:3": "，比例 「2:3」",
    "3:4": "，比例 「3:4」",
    "4:3": "，比例 「4:3」",
    "9:16": "，比例 「9:16」",
    "16:9": "，比例 「16:9」",
}


class DoubaoBasePlaywrightNode:
    CATEGORY = "Doubao/Playwright"
    FUNCTION = "run"
    OUTPUT_NODE = True

    NEW_CONVERSATION = False
    TRY_HD_DOWNLOAD = True
    BROWSER_EXECUTABLE_PATH = ""
    USER_DATA_DIR = ""
    INPUT_SELECTOR = ""
    UPLOAD_SELECTOR = ""
    SEND_BUTTON_SELECTOR = ""
    NEW_CHAT_SELECTOR = ""
    ASSISTANT_MESSAGE_SELECTOR = ""

    @classmethod
    def _common_inputs(cls) -> dict[str, Any]:
        return {
            "website_url": (
                "STRING",
                {
                    "default": "https://www.doubao.com/chat/",
                },
            ),
            "image_save_path": (
                "STRING",
                {
                    "default": "output/doubao_playwright",
                },
            ),
            "browser_channel": (
                ["chromium", "chrome", "msedge"],
                {
                    "default": "chromium",
                },
            ),
            "wait_timeout": (
                "INT",
                {
                    "default": 240,
                    "min": 30,
                    "max": 1800,
                    "step": 10,
                },
            ),
            "stable_seconds": (
                "INT",
                {
                    "default": 4,
                    "min": 2,
                    "max": 60,
                    "step": 1,
                },
            ),
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs: Any) -> float:
        return time.time()

    def _send_to_doubao(
        self,
        *,
        prompt_text: str,
        image_tensors: list[torch.Tensor | None] | None = None,
        collect_images: bool,
        min_output_images: int,
        website_url: str,
        image_save_path: str,
        browser_channel: str,
        wait_timeout: int,
        stable_seconds: int,
    ) -> tuple[list[Path], str]:
        save_dir = _resolve_path(image_save_path, "output/doubao_playwright")
        save_dir.mkdir(parents=True, exist_ok=True)
        input_paths = self._save_input_images(save_dir, image_tensors or [])

        return SESSION.send_and_collect(
            website_url=website_url,
            image_paths=input_paths,
            prompt_text=prompt_text or "",
            save_dir=save_dir,
            new_conversation=self.NEW_CONVERSATION,
            browser_channel=browser_channel,
            browser_executable_path=self.BROWSER_EXECUTABLE_PATH,
            user_data_dir=self.USER_DATA_DIR,
            input_selector=self.INPUT_SELECTOR,
            upload_selector=self.UPLOAD_SELECTOR,
            send_button_selector=self.SEND_BUTTON_SELECTOR,
            new_chat_selector=self.NEW_CHAT_SELECTOR,
            assistant_message_selector=self.ASSISTANT_MESSAGE_SELECTOR,
            wait_timeout=wait_timeout,
            stable_seconds=stable_seconds,
            min_output_images=min_output_images,
            try_hd_download=self.TRY_HD_DOWNLOAD,
            collect_images=collect_images,
        )

    @staticmethod
    def _compose_image_prompt(prompt_prefix: str, text: str, image_size: str) -> str:
        prefix = prompt_prefix if prompt_prefix is not None else "帮我生成图片："
        suffix = RATIO_SUFFIXES.get(image_size or "auto", "")
        return f"{prefix}{text or ''}{suffix}"

    @staticmethod
    def _image_output(downloaded: list[Path]) -> torch.Tensor:
        output_images = _paths_to_image_batch(downloaded[:4])
        if output_images is None:
            return _blank_image()
        _log(f"图片输出 batch 数量：{output_images.shape[0]}")
        return output_images

    def _save_input_images(self, save_dir: Path, images: list[torch.Tensor | None]) -> list[Path]:
        upload_dir = save_dir / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        run_id = time.strftime("%Y%m%d_%H%M%S")
        paths: list[Path] = []
        for port_index, image in enumerate(images, start=1):
            if image is None:
                continue
            frames = image if image.ndim == 4 else image[None, ...]
            for batch_index, frame in enumerate(frames):
                if len(paths) >= 5:
                    return paths
                pil_image = _tensor_frame_to_pil(frame)
                path = upload_dir / f"input_{run_id}_{port_index}_{batch_index}.png"
                pil_image.save(path)
                paths.append(path)
        return paths


class DoubaoTextToTextNode(DoubaoBasePlaywrightNode):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                **cls._common_inputs(),
            },
        }

    def run(
        self,
        text: str,
        website_url: str,
        image_save_path: str,
        browser_channel: str,
        wait_timeout: int,
        stable_seconds: int,
    ) -> dict[str, Any]:
        _, generated_text = self._send_to_doubao(
            prompt_text=text or "",
            collect_images=False,
            min_output_images=0,
            website_url=website_url,
            image_save_path=image_save_path,
            browser_channel=browser_channel,
            wait_timeout=int(wait_timeout),
            stable_seconds=int(stable_seconds),
        )
        return {
            "ui": {"text": [generated_text]},
            "result": (generated_text,),
        }


class DoubaoTextToImageNode(DoubaoBasePlaywrightNode):
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "prompt_prefix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "帮我生成图片：",
                    },
                ),
                "image_size": (list(RATIO_SUFFIXES.keys()), {"default": "auto"}),
                **cls._common_inputs(),
            },
        }

    def run(
        self,
        text: str,
        prompt_prefix: str,
        image_size: str,
        website_url: str,
        image_save_path: str,
        browser_channel: str,
        wait_timeout: int,
        stable_seconds: int,
    ) -> dict[str, Any]:
        prompt_text = self._compose_image_prompt(prompt_prefix, text, image_size)
        downloaded, generated_text = self._send_to_doubao(
            prompt_text=prompt_text,
            collect_images=True,
            min_output_images=1,
            website_url=website_url,
            image_save_path=image_save_path,
            browser_channel=browser_channel,
            wait_timeout=int(wait_timeout),
            stable_seconds=int(stable_seconds),
        )
        return {
            "ui": {
                "text": [generated_text],
                "saved_images": [str(path) for path in downloaded],
            },
            "result": (self._image_output(downloaded), generated_text),
        }


class DoubaoImageToTextNode(DoubaoBasePlaywrightNode):
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt_prefix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "详细描述图片，",
                    },
                ),
                **cls._common_inputs(),
            },
        }

    def run(
        self,
        image: torch.Tensor,
        prompt_prefix: str,
        website_url: str,
        image_save_path: str,
        browser_channel: str,
        wait_timeout: int,
        stable_seconds: int,
    ) -> dict[str, Any]:
        _, generated_text = self._send_to_doubao(
            prompt_text=prompt_prefix or "",
            image_tensors=[image],
            collect_images=False,
            min_output_images=0,
            website_url=website_url,
            image_save_path=image_save_path,
            browser_channel=browser_channel,
            wait_timeout=int(wait_timeout),
            stable_seconds=int(stable_seconds),
        )
        return {
            "ui": {"text": [generated_text]},
            "result": (generated_text,),
        }


class DoubaoImagesTextToImageNode(DoubaoBasePlaywrightNode):
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                    },
                ),
                "prompt_prefix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "帮我生成图片：",
                    },
                ),
                "image_size": (list(RATIO_SUFFIXES.keys()), {"default": "auto"}),
                **cls._common_inputs(),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }

    def run(
        self,
        text: str,
        prompt_prefix: str,
        image_size: str,
        website_url: str,
        image_save_path: str,
        browser_channel: str,
        wait_timeout: int,
        stable_seconds: int,
        image_1: torch.Tensor | None = None,
        image_2: torch.Tensor | None = None,
        image_3: torch.Tensor | None = None,
        image_4: torch.Tensor | None = None,
        image_5: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        prompt_text = self._compose_image_prompt(prompt_prefix, text, image_size)
        downloaded, generated_text = self._send_to_doubao(
            prompt_text=prompt_text,
            image_tensors=[image_1, image_2, image_3, image_4, image_5],
            collect_images=True,
            min_output_images=1,
            website_url=website_url,
            image_save_path=image_save_path,
            browser_channel=browser_channel,
            wait_timeout=int(wait_timeout),
            stable_seconds=int(stable_seconds),
        )
        return {
            "ui": {
                "text": [generated_text],
                "saved_images": [str(path) for path in downloaded],
            },
            "result": (self._image_output(downloaded), generated_text),
        }


NODE_CLASS_MAPPINGS = {
    "DoubaoTextToTextNode": DoubaoTextToTextNode,
    "DoubaoTextToImageNode": DoubaoTextToImageNode,
    "DoubaoImageToTextNode": DoubaoImageToTextNode,
    "DoubaoImagesTextToImageNode": DoubaoImagesTextToImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DoubaoTextToTextNode": "Doubao 文生文",
    "DoubaoTextToImageNode": "Doubao 文生图",
    "DoubaoImageToTextNode": "Doubao 图生文",
    "DoubaoImagesTextToImageNode": "Doubao 图文生图",
}

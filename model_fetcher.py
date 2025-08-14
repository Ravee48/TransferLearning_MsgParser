import os
import re
import shutil
import tempfile
import zipfile
from typing import List

import requests


class ModelFetcher:
    """
    Ensures a minimal Hugging Face model folder exists locally by optionally
    downloading and extracting a zip archive from Google Drive (public) or a
    direct URL.

    Uses environment variables by default:
      - MODEL_ZIP_URL: direct URL to a zip containing the model files
      - GDRIVE_FILE_ID: Google Drive file id; used if MODEL_ZIP_URL not set
    """

    REQUIRED_FILES: List[str] = [
        # Minimal required manifest; validated via has_required_files
        "config.json",
    ]

    def __init__(self, model_zip_url: str | None = None, gdrive_file_id: str | None = None) -> None:
        # Prefer explicit constructor args; fall back to environment variables (case-insensitive for convenience)
        self.model_zip_url = model_zip_url or self._get_env_any(
            "MODEL_ZIP_URL", "Model_Zip_Url", "MODEL_URL"
        )
        explicit_id = gdrive_file_id or self._get_env_any(
            "GDRIVE_FILE_ID", "GDRIVE_FILE_Id", "GDRIVE_FILEID", "GOOGLE_DRIVE_FILE_ID"
        )
        self.gdrive_file_id = explicit_id

        # Allow passing a full Drive URL instead of an id
        if not self.gdrive_file_id:
            gdrive_url = self._get_env_any("GDRIVE_FILE_URL", "GDRIVE_URL", "GOOGLE_DRIVE_URL")
            if gdrive_url:
                maybe_id = self._extract_gdrive_id(gdrive_url)
                if maybe_id:
                    self.gdrive_file_id = maybe_id

    def has_required_files(self, directory: str) -> bool:
        if not os.path.isdir(directory):
            return False
        # Must have config and one of the weight files
        has_config = os.path.isfile(os.path.join(directory, "config.json"))
        has_weights = os.path.isfile(os.path.join(directory, "model.safetensors")) or os.path.isfile(
            os.path.join(directory, "pytorch_model.bin")
        )
        # Must have at least one tokenizer artifact
        has_tokenizer = os.path.isfile(os.path.join(directory, "tokenizer.json")) or os.path.isfile(
            os.path.join(directory, "vocab.txt")
        )
        return has_config and has_weights and has_tokenizer

    def ensure_model_present(self, target_dir: str) -> None:
        os.makedirs(target_dir, exist_ok=True)
        if self.has_required_files(target_dir):
            return

        archive_url = self._resolve_archive_url()
        if not archive_url:
            raise RuntimeError(
                "MODEL_ZIP_URL or GDRIVE_FILE_ID must be set to fetch the model archive when the model directory is empty."
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "model.zip")
            self._download_with_google_drive_support(archive_url, zip_path)
            if not zipfile.is_zipfile(zip_path):
                raise RuntimeError(
                    "Downloaded file is not a ZIP archive. Please upload/share a .zip containing the model folder."
                )
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
            candidate = self._find_model_subdir(tmpdir)
            if candidate is None:
                raise RuntimeError("Downloaded archive does not contain required model files")
            self._copy_tree(candidate, target_dir)
            if not self.has_required_files(target_dir):
                raise RuntimeError("Model files missing after extraction. Check archive contents.")

    def _resolve_archive_url(self) -> str | None:
        # If a direct URL is provided and it's not a Google Drive share link, return as-is.
        if self.model_zip_url:
            # If the provided URL looks like a Google Drive share URL, convert it to a direct download URL.
            if "drive.google.com" in self.model_zip_url:
                extracted = self._extract_gdrive_id(self.model_zip_url)
                if extracted:
                    return f"https://drive.google.com/uc?export=download&id={extracted}"
            return self.model_zip_url
        if self.gdrive_file_id:
            return f"https://drive.google.com/uc?export=download&id={self.gdrive_file_id}"
        return None

    def _extract_gdrive_id(self, url: str) -> str | None:
        # Handle URLs like: https://drive.google.com/file/d/<id>/view?usp=sharing
        m = re.search(r"/file/d/([0-9A-Za-z_-]{20,})/", url)
        if m:
            return m.group(1)
        # Handle URLs like: https://drive.google.com/open?id=<id> or ...?id=<id>
        m = re.search(r"[?&]id=([0-9A-Za-z_-]{20,})", url)
        if m:
            return m.group(1)
        return None

    def _find_model_subdir(self, root: str) -> str | None:
        if self.has_required_files(root):
            return root
        for dirpath, dirnames, filenames in os.walk(root):
            if self.has_required_files(dirpath):
                return dirpath
        return None

    def _copy_tree(self, src: str, dst: str) -> None:
        for name in os.listdir(src):
            s = os.path.join(src, name)
            d = os.path.join(dst, name)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)
                self._copy_tree(s, d)
            else:
                shutil.copy2(s, d)

    def _download_with_google_drive_support(self, url: str, dst_path: str) -> None:
        session = requests.Session()
        # Special handling for Google Drive share/uc URLs
        if "drive.google.com" in url:
            file_id = self._extract_gdrive_id(url)
            if file_id:
                self._download_from_gdrive_id(session, file_id, dst_path)
                return

        # Fallback: regular HTTP(S) download
        response = session.get(url, stream=True, allow_redirects=True)
        self._save_response_content(response, dst_path)

        # If we downloaded HTML from Drive by accident, try additional fallbacks
        if not self._is_zip_file(dst_path) and ("drive.google.com" in url or "drive.usercontent.google.com" in url):
            file_id = self._extract_gdrive_id(url)
            if file_id:
                self._download_from_gdrive_id(session, file_id, dst_path)
                if not self._is_zip_file(dst_path):
                    self._try_gdown(file_id, dst_path)

    def _get_confirm_token(self, response: requests.Response) -> str | None:
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        try:
            txt = response.text
            # Token can include hyphens as well
            m = re.search(r"confirm=([0-9A-Za-z_-]+)", txt)
            if m:
                return m.group(1)
            m = re.search(r"data-confirm-token=\"([^\"]+)\"", txt)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    def _save_response_content(self, response: requests.Response, destination: str, chunk_size: int = 32768) -> None:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        # If Content-Type hints it's HTML, leave further handling to caller

    def _is_zip_file(self, path: str) -> bool:
        try:
            if zipfile.is_zipfile(path):
                return True
            # Additional heuristic: check for zip magic number
            with open(path, "rb") as f:
                sig = f.read(4)
            return sig == b"PK\x03\x04"
        except Exception:
            return False

    def _download_from_gdrive_id(self, session: requests.Session, file_id: str, dst_path: str) -> None:
        # Try the usercontent host first, it usually bypasses the interstitial
        base_usercontent = "https://drive.usercontent.google.com/download"
        params = {"id": file_id, "export": "download"}
        response = session.get(base_usercontent, params=params, stream=True, allow_redirects=True)

        # If Drive returns an HTML page, we may need a confirm token
        token = self._get_confirm_token(response)
        if token:
            params["confirm"] = token
            response = session.get(base_usercontent, params=params, stream=True, allow_redirects=True)

        # Save initial attempt
        self._save_response_content(response, dst_path)

        # If still not a zip, try the uc endpoint as a fallback
        if not zipfile.is_zipfile(dst_path):
            uc_url = "https://drive.google.com/uc"
            params = {"id": file_id, "export": "download"}
            response = session.get(uc_url, params=params, stream=True, allow_redirects=True)
            token = self._get_confirm_token(response)
            if token:
                params["confirm"] = token
                response = session.get(uc_url, params=params, stream=True, allow_redirects=True)
            self._save_response_content(response, dst_path)
        if not self._is_zip_file(dst_path):
            self._try_gdown(file_id, dst_path)

    def _try_gdown(self, file_id: str, dst_path: str) -> None:
        try:
            import gdown  # type: ignore
        except Exception as e:
            return  # gdown not available; caller will error with non-zip
        # gdown can handle Drive confirmations more robustly
        url = f"https://drive.google.com/uc?id={file_id}"
        tmp_path = dst_path + ".tmp"
        gdown.download(url=url, output=tmp_path, quiet=True, fuzzy=True, use_cookies=False)
        if os.path.exists(tmp_path):
            shutil.move(tmp_path, dst_path)

    def _get_env_any(self, *names: str) -> str | None:
        for name in names:
            # direct
            if name in os.environ:
                return os.environ.get(name)
            # case-insensitive search
            for k, v in os.environ.items():
                if k.lower() == name.lower():
                    return v
        return None



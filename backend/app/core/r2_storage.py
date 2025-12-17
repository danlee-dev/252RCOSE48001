"""
Cloudflare R2 Storage Service
- S3-compatible object storage
- Upload/download files for distributed services (Backend, Celery)
"""

import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Optional, BinaryIO
import tempfile
from pathlib import Path


class R2Storage:
    """Cloudflare R2 Storage Client"""

    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID")
        self.access_key_id = os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = os.getenv("R2_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("R2_BUCKET_NAME", "docscanner-contracts")

        self._client = None
        self._enabled = all([
            self.account_id,
            self.access_key_id,
            self.secret_access_key
        ])

    @property
    def enabled(self) -> bool:
        """Check if R2 is configured"""
        return self._enabled

    @property
    def client(self):
        """Lazy initialization of S3 client"""
        if self._client is None and self._enabled:
            self._client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"}
                )
            )
        return self._client

    def upload_file(
        self,
        file_content: bytes,
        object_key: str,
        content_type: str = "application/octet-stream"
    ) -> Optional[str]:
        """
        Upload file to R2

        Args:
            file_content: File bytes
            object_key: S3 object key (e.g., "contracts/1/file.pdf")
            content_type: MIME type

        Returns:
            Object key if successful, None otherwise
        """
        if not self.enabled:
            print("[R2] Storage not configured, skipping upload")
            return None

        try:
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=file_content,
                ContentType=content_type
            )
            print(f"[R2] Uploaded: {object_key}")
            return object_key
        except ClientError as e:
            print(f"[R2] Upload failed: {e}")
            return None

    def download_file(self, object_key: str) -> Optional[bytes]:
        """
        Download file from R2

        Args:
            object_key: S3 object key

        Returns:
            File bytes if successful, None otherwise
        """
        if not self.enabled:
            print("[R2] Storage not configured, skipping download")
            return None

        try:
            response = self.client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            content = response["Body"].read()
            print(f"[R2] Downloaded: {object_key} ({len(content)} bytes)")
            return content
        except ClientError as e:
            print(f"[R2] Download failed: {e}")
            return None

    def download_to_temp_file(self, object_key: str) -> Optional[str]:
        """
        Download file from R2 to a temporary file

        Args:
            object_key: S3 object key

        Returns:
            Temporary file path if successful, None otherwise
        """
        content = self.download_file(object_key)
        if content is None:
            return None

        # Get file extension from object key
        extension = Path(object_key).suffix or ".tmp"

        # Create temp file with proper extension
        fd, temp_path = tempfile.mkstemp(suffix=extension)
        try:
            os.write(fd, content)
            os.close(fd)
            print(f"[R2] Saved to temp: {temp_path}")
            return temp_path
        except Exception as e:
            os.close(fd)
            os.unlink(temp_path)
            print(f"[R2] Failed to save temp file: {e}")
            return None

    def delete_file(self, object_key: str) -> bool:
        """
        Delete file from R2

        Args:
            object_key: S3 object key

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            self.client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            print(f"[R2] Deleted: {object_key}")
            return True
        except ClientError as e:
            print(f"[R2] Delete failed: {e}")
            return False

    def file_exists(self, object_key: str) -> bool:
        """Check if file exists in R2"""
        if not self.enabled:
            return False

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True
        except ClientError:
            return False

    def get_public_url(self, object_key: str) -> Optional[str]:
        """
        Get public URL for file (requires public bucket or custom domain)

        Args:
            object_key: S3 object key

        Returns:
            Public URL string
        """
        public_url = os.getenv("R2_PUBLIC_URL")
        if public_url:
            return f"{public_url.rstrip('/')}/{object_key}"
        return None


# Singleton instance
r2_storage = R2Storage()


def get_r2_object_key(user_id: int, filename: str) -> str:
    """Generate R2 object key from user ID and filename"""
    return f"contracts/{user_id}/{filename}"


def parse_r2_object_key(file_url: str) -> Optional[str]:
    """
    Parse file_url to extract R2 object key

    Handles both formats:
    - R2 key: "contracts/1/file.pdf"
    - Legacy local path: "/storage/contracts/1/file.pdf"
    """
    if file_url.startswith("contracts/"):
        return file_url

    if "/storage/contracts/" in file_url:
        # Extract "contracts/1/file.pdf" from "/storage/contracts/1/file.pdf"
        return file_url.split("/storage/")[-1]

    return None

"""S3 storage adapter implementation."""

import logging
from typing import Any
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from botocore.config import Config

from .base import StorageError

logger = logging.getLogger(__name__)


class S3Adapter:
    """AWS S3 storage adapter.

    Implements the StorageAdapter interface for S3 operations.
    Supports both authenticated and anonymous access.
    """

    def __init__(
        self,
        region_name: str = "us-west-2",
        use_ssl: bool = True,
        anonymous: bool = False,
        max_retries: int = 3,
    ):
        """Initialize S3 adapter.

        Args:
            region_name: AWS region name
            use_ssl: Whether to use SSL for connections
            anonymous: Whether to use anonymous access (no credentials)
            max_retries: Maximum number of retry attempts
        """
        self.region_name = region_name
        self.use_ssl = use_ssl
        self.anonymous = anonymous

        # Configure retry strategy
        config = Config(
            region_name=region_name,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
        )

        # Create client
        if anonymous:
            from botocore import UNSIGNED
            from botocore.config import Config as BotoConfig

            config = config.merge(BotoConfig(signature_version=UNSIGNED))
            self._client = boto3.client("s3", config=config, use_ssl=use_ssl)
        else:
            self._client = boto3.client(
                "s3", region_name=region_name, config=config, use_ssl=use_ssl
            )

    def download(self, bucket: str, key: str, local_path: Path | None = None) -> bytes:
        """Download object from S3.

        Args:
            bucket: S3 bucket name
            key: Object key
            local_path: Optional local path to save file

        Returns:
            Object contents as bytes

        Raises:
            StorageError: If download fails
        """
        try:
            if local_path:
                # Download to file
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._client.download_file(bucket, key, str(local_path))
                logger.debug(f"Downloaded s3://{bucket}/{key} to {local_path}")
                return local_path.read_bytes()
            else:
                # Download to memory
                response = self._client.get_object(Bucket=bucket, Key=key)
                data = response["Body"].read()
                logger.debug(f"Downloaded s3://{bucket}/{key} ({len(data)} bytes)")
                return data

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                raise StorageError(f"Object not found: s3://{bucket}/{key}")
            elif error_code == "AccessDenied":
                raise StorageError(f"Access denied: s3://{bucket}/{key}")
            else:
                raise StorageError(f"S3 download failed: {e}")
        except NoCredentialsError:
            raise StorageError("No AWS credentials found")
        except Exception as e:
            raise StorageError(f"Unexpected error downloading from S3: {e}")

    def upload(self, bucket: str, key: str, data: bytes) -> str:
        """Upload data to S3.

        Args:
            bucket: S3 bucket name
            key: Object key
            data: Data to upload

        Returns:
            S3 URI of uploaded object

        Raises:
            StorageError: If upload fails
        """
        if self.anonymous:
            raise StorageError("Cannot upload with anonymous access")

        try:
            self._client.put_object(Bucket=bucket, Key=key, Body=data)
            logger.debug(f"Uploaded {len(data)} bytes to s3://{bucket}/{key}")
            return f"s3://{bucket}/{key}"

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "AccessDenied":
                raise StorageError(f"Access denied: s3://{bucket}/{key}")
            else:
                raise StorageError(f"S3 upload failed: {e}")
        except NoCredentialsError:
            raise StorageError("No AWS credentials found")
        except Exception as e:
            raise StorageError(f"Unexpected error uploading to S3: {e}")

    def exists(self, bucket: str, key: str) -> bool:
        """Check if object exists in S3.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            True if object exists
        """
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            # For other errors, we can't determine existence
            logger.warning(f"Error checking existence of s3://{bucket}/{key}: {e}")
            return False
        except Exception:
            return False

    def list_objects(self, bucket: str, prefix: str = "") -> list[str]:
        """List objects in S3 bucket with given prefix.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix to filter objects

        Returns:
            List of object keys
        """
        try:
            keys = []
            paginator = self._client.get_paginator("list_objects_v2")

            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if "Contents" in page:
                    keys.extend([obj["Key"] for obj in page["Contents"]])

            logger.debug(f"Listed {len(keys)} objects in s3://{bucket}/{prefix}")
            return keys

        except ClientError as e:
            raise StorageError(f"Failed to list objects: {e}")
        except Exception as e:
            raise StorageError(f"Unexpected error listing objects: {e}")

    def download_file(self, bucket: str, key: str, local_path: Path) -> None:
        """Download file directly to disk (more efficient for large files).

        Args:
            bucket: S3 bucket name
            key: Object key
            local_path: Local path to save file

        Raises:
            StorageError: If download fails
        """
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(bucket, key, str(local_path))
            logger.debug(f"Downloaded s3://{bucket}/{key} to {local_path}")

        except ClientError as e:
            raise StorageError(f"S3 download failed: {e}")
        except Exception as e:
            raise StorageError(f"Unexpected error downloading from S3: {e}")

    def upload_file(self, local_path: Path, bucket: str, key: str) -> str:
        """Upload file from disk (more efficient for large files).

        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: Object key

        Returns:
            S3 URI of uploaded object

        Raises:
            StorageError: If upload fails
        """
        if self.anonymous:
            raise StorageError("Cannot upload with anonymous access")

        try:
            self._client.upload_file(str(local_path), bucket, key)
            logger.debug(f"Uploaded {local_path} to s3://{bucket}/{key}")
            return f"s3://{bucket}/{key}"

        except ClientError as e:
            raise StorageError(f"S3 upload failed: {e}")
        except Exception as e:
            raise StorageError(f"Unexpected error uploading to S3: {e}")

    def get_object_info(self, bucket: str, key: str) -> dict[str, Any]:
        """Get object metadata without downloading.

        Args:
            bucket: S3 bucket name
            key: Object key

        Returns:
            Dictionary with object metadata

        Raises:
            StorageError: If operation fails
        """
        try:
            response = self._client.head_object(Bucket=bucket, Key=key)
            return {
                "size": response.get("ContentLength", 0),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
                "content_type": response.get("ContentType", ""),
                "metadata": response.get("Metadata", {}),
            }
        except ClientError as e:
            raise StorageError(f"Failed to get object info: {e}")
        except Exception as e:
            raise StorageError(f"Unexpected error getting object info: {e}")

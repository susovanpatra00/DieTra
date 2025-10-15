import os
from pathlib import Path
from typing import Optional, List, Dict
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from config import DEFAULT_S3_REGION

load_dotenv()


class CloudUploader:
    """Upload and manage files in AWS S3."""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = DEFAULT_S3_REGION
    ):
        self.bucket_name = bucket_name

        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key

        self.s3_client = boto3.client("s3", **session_kwargs)

        # Verify bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ Connected to S3 bucket: {bucket_name}")
        except ClientError as e:
            raise RuntimeError(f"Failed to access S3 bucket {bucket_name}: {e}")

    # -------------------------------------------------------------------------
    # Upload
    # -------------------------------------------------------------------------

    def upload_file(self, local_path: str, s3_key: str, extra_args: Optional[Dict] = None) -> str:
        """Upload a single file to S3."""
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key, ExtraArgs=extra_args or {})
            return f"s3://{self.bucket_name}/{s3_key}"
        except ClientError as e:
            raise RuntimeError(f"Failed to upload {local_path} to S3: {e}")

    def upload_directory(self, local_dir: str, s3_prefix: str, file_extensions: Optional[List[str]] = None) -> List[Dict]:
        """Upload all files in a directory to S3."""
        uploaded = []
        local_dir_path = Path(local_dir)

        for root, _, files in os.walk(local_dir):
            for file in files:
                if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                    continue

                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_dir_path)
                s3_key = str(Path(s3_prefix) / relative_path).replace("\\", "/")

                try:
                    s3_uri = self.upload_file(str(local_path), s3_key)
                    uploaded.append({"local_path": str(local_path), "s3_key": s3_key, "s3_uri": s3_uri, "filename": file})
                    print(f"  ✓ Uploaded: {file} -> {s3_uri}")
                except Exception as e:
                    print(f"  ✗ Failed: {file} - {e}")

        return uploaded

    # -------------------------------------------------------------------------
    # List / Delete
    # -------------------------------------------------------------------------

    def list_files(self, prefix: str = "") -> List[str]:
        """List all files in the bucket optionally filtered by prefix."""
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            files.extend([obj["Key"] for obj in page.get("Contents", [])])
        return files

    def list_folders(self, prefix: str = "") -> List[str]:
        """List folders in the bucket under a prefix."""
        folders = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix, Delimiter="/"):
            folders.extend([p["Prefix"] for p in page.get("CommonPrefixes", [])])
        return folders

    def delete_all_files(self, prefix: str = ""):
        """Delete all files in the bucket (optionally under a prefix)."""
        print(f"⚠ Deleting all files in bucket '{self.bucket_name}' with prefix '{prefix}'...")
        paginator = self.s3_client.get_paginator("list_objects_v2")
        delete_batch = {"Objects": []}
        count = 0

        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                delete_batch["Objects"].append({"Key": obj["Key"]})
                count += 1
                if len(delete_batch["Objects"]) >= 1000:
                    self.s3_client.delete_objects(Bucket=self.bucket_name, Delete=delete_batch)
                    delete_batch["Objects"] = []

        if delete_batch["Objects"]:
            self.s3_client.delete_objects(Bucket=self.bucket_name, Delete=delete_batch)

        print(f"✓ Deleted {count} files from bucket '{self.bucket_name}'")


# -------------------------------------------------------------------------
# CLI / Test
# -------------------------------------------------------------------------

if __name__ == "__main__":
    S3_BUCKET = os.getenv("S3_BUCKET")
    AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")

    uploader = CloudUploader(bucket_name=S3_BUCKET, aws_access_key_id=AWS_KEY, aws_secret_access_key=AWS_SECRET)

    # List files
    all_files = uploader.list_files()
    print(f"Total files: {len(all_files)}")
    for f in all_files:
        print(f"- {f}")

    # Confirm and delete all files
    confirm = input("\nAre you sure you want to delete ALL files from this bucket? (yes/no): ")
    if confirm.lower() == "yes":
        uploader.delete_all_files()
    else:
        print("Aborted — no files deleted.")

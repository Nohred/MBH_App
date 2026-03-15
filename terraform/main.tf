terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region                      = "us-east-1"
  access_key                  = "test"
  secret_key                  = "test"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3  = "http://localhost:4566"
    iam = "http://localhost:4566"
  }
}

# 1. S3 BUCKET
resource "aws_s3_bucket" "my_vite_app" {
  bucket = "my-vite-app-bucket"
}

# 2. TURN OFF PUBLIC ACCESS BLOCK
# We must turn this off so the public internet can view the website files
resource "aws_s3_bucket_public_access_block" "allow_public" {
  bucket                  = aws_s3_bucket.my_vite_app.id
  block_public_acls       = false
  block_public_policy     = false
  ignore_public_acls      = false
  restrict_public_buckets = false
}

# 3. ENABLE S3 WEBSITE HOSTING
resource "aws_s3_bucket_website_configuration" "website" {
  bucket = aws_s3_bucket.my_vite_app.id
  index_document {
    suffix = "index.html"
  }
}

# 4. IAM BUCKET POLICY (The rule that lets visitors see your site)
resource "aws_s3_bucket_policy" "public_read" {
  bucket = aws_s3_bucket.my_vite_app.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = "*"
        Action    = "s3:GetObject"
        Resource  = "${aws_s3_bucket.my_vite_app.arn}/*"
      }
    ]
  })
  
  # Terraform must wait to unblock public access before applying this policy
  depends_on = [aws_s3_bucket_public_access_block.allow_public]
}

# 5. IAM USER (Your deployer bot)
resource "aws_iam_user" "deploy_user" {
  name = "vite-app-deployer"
}

# 6. IAM USER POLICY (Least Privilege - only allowed to upload to this bucket)
resource "aws_iam_user_policy" "deploy_policy" {
  name = "S3DeployPolicy"
  user = aws_iam_user.deploy_user.name
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.my_vite_app.arn,
          "${aws_s3_bucket.my_vite_app.arn}/*"
        ]
      }
    ]
  })
}

terraform {
  backend "s3" {
    bucket = "tfstate-k8s-training-0fba25e3a3f1ddc4189c936b75f38e24"
    key    = "k8s-training.tfstate"

    endpoints = {
      s3 = "https://storage.me-west1.nebius.cloud:443"
    }
    region = "me-west1"

    skip_region_validation      = true
    skip_credentials_validation = true
    skip_requesting_account_id  = true
    skip_s3_checksum            = true
  }
}

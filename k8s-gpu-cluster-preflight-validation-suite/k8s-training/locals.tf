locals {
  release-suffix = random_string.random.result
  ssh_public_key = var.ssh_public_key.key != null ? var.ssh_public_key.key : (
  fileexists(var.ssh_public_key.path) ? file(var.ssh_public_key.path) : null)

  filestore = {
    mount_tag = "data"
  }

  regions_default = {
    eu-west1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "fabric-5"
    }
    eu-north1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h100-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "fabric-3"
    }
    eu-north2 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "eu-north2-a"
    }
    us-central1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-h200-sxm"
      gpu_nodes_preset   = "8gpu-128vcpu-1600gb"
      infiniband_fabric  = "us-central1-a"
    }
    me-west1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-b200-sxm-a"
      gpu_nodes_preset   = "8gpu-160vcpu-1792gb"
      infiniband_fabric  = "ramon"
    }
    uk-south1 = {
      cpu_nodes_platform = "cpu-d3"
      cpu_nodes_preset   = "16vcpu-64gb"
      gpu_nodes_platform = "gpu-b300-sxm"
      gpu_nodes_preset   = "8gpu-192vcpu-2768gb"
      infiniband_fabric  = "uk-south1-a"
    }
  }

  current_region_defaults = local.regions_default[var.region]

  cpu_nodes_preset   = coalesce(var.cpu_nodes_preset, local.current_region_defaults.cpu_nodes_preset)
  cpu_nodes_platform = coalesce(var.cpu_nodes_platform, local.current_region_defaults.cpu_nodes_platform)
  gpu_nodes_platform = coalesce(var.gpu_nodes_platform, local.current_region_defaults.gpu_nodes_platform)
  gpu_nodes_preset   = coalesce(var.gpu_nodes_preset, local.current_region_defaults.gpu_nodes_preset)
  infiniband_fabric  = coalesce(var.infiniband_fabric, local.current_region_defaults.infiniband_fabric)

  platform_to_cuda = {
    gpu-b200-sxm-a = "cuda12.8"
    gpu-b300-sxm   = "cuda13.0"
  }
  device_preset = lookup(local.platform_to_cuda, local.gpu_nodes_platform, "cuda12")

  valid_mig_parted_configs = {
    "gpu-h100-sxm"   = ["all-disabled", "all-enabled", "all-balanced", "all-1g.10gb", "all-1g.10gb.me", "all-1g.20gb", "all-2g.20gb", "all-3g.40gb", "all-4g.40gb", "all-7g.80gb"]
    "gpu-h200-sxm"   = ["all-disabled", "all-enabled", "all-balanced", "all-1g.18gb", "all-1g.18gb.me", "all-1g.35gb", "all-2g.35gb", "all-3g.71gb", "all-4g.71gb", "all-7g.141gb"]
    "gpu-b200-sxm"   = ["all-disabled", "all-enabled", "all-balanced", "all-1g.23gb", "all-1g.23gb.me", "all-1g.45gb", "all-2g.45gb", "all-3g.90gb", "all-4g.90gb", "all-7g.180gb"]
    "gpu-b200-sxm-a" = ["all-disabled", "all-enabled", "all-balanced", "all-1g.23gb", "all-1g.23gb.me", "all-1g.45gb", "all-2g.45gb", "all-3g.90gb", "all-4g.90gb", "all-7g.180gb"]
    "gpu-b300-sxm"   = ["all-disabled", "all-enabled", "all-balanced", "all-1g.23gb", "all-1g.23gb.me", "all-1g.45gb", "all-2g.45gb", "all-3g.90gb", "all-4g.90gb", "all-7g.180gb"]

  }

  # Mapping from platform and preset to hardware profile for nebius-gpu-health-checker
  platform_preset_to_hardware_profile = {
    # H100 configurations
    "gpu-h100-sxm-1gpu-16vcpu-200gb"   = "1xH100"
    "gpu-h100-sxm-8gpu-128vcpu-1600gb" = "8xH100"

    # H200 configurations
    "gpu-h200-sxm-1gpu-16vcpu-200gb"   = "1xH200"
    "gpu-h200-sxm-8gpu-128vcpu-1600gb" = "8xH200"

    # B200 configurations
    "gpu-b200-sxm-1gpu-20vcpu-224gb"     = "1xB200"
    "gpu-b200-sxm-8gpu-160vcpu-1792gb"   = "8xB200"
    "gpu-b200-sxm-a-8gpu-160vcpu-1792gb" = "8xB200"

    #B300 configuration
    "gpu-b300-sxm-8gpu-192vcpu-2768gb" = "8xB300"
    "gpu-b300-sxm-1gpu-24vcpu-346gb"   = "1xB300"

    # L40 configurations
    "gpu-l40s-d-1gpu-16vcpu-96gb"    = "1xL40S"
    "gpu-l40s-d-1gpu-32vcpu-192gb"   = "1xL40S"
    "gpu-l40s-d-1gpu-48vcpu-288gb"   = "1xL40S"
    "gpu-l40s-d-2gpu-64vcpu-384gb"   = "2xL40S"
    "gpu-l40s-d-2gpu-64vcpu-384gb"   = "2xL40S"
    "gpu-l40s-d-2gpu-96vcpu-576gb"   = "2xL40S"
    "gpu-l40s-d-4gpu-128vcpu-768gb"  = "4xL40S"
    "gpu-l40s-d-4gpu-192vcpu-1152gb" = "4xL40S"
    "gpu-l40s-a-1gpu-8vcpu-32gb"     = "1XL40S"
    "gpu-l40s-a-1gpu-24vcpu-96gb"    = "1X40S"
    "gpu-l40s-a-1gpu-32vcpu-128gb"   = "1X40S"
    "gpu-l40s-a-1gpu-40vcpu-160gb"   = "1X40S"
  }

  # Create the key for hardware profile lookup
  hardware_profile_key = "${local.gpu_nodes_platform}-${local.gpu_nodes_preset}"
}

resource "random_string" "random" {
  keepers = {
    ami_id = "${var.parent_id}"
  }
  length  = 6
  upper   = true
  lower   = true
  numeric = true
  special = false
}

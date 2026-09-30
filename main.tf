terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {}

# Reference the BoltGuard AI image already built locally (via `docker build`
# earlier), instead of having Terraform rebuild it itself.
data "docker_image" "boltguard" {
  name = "boltguard-ai:latest"
}

# Run a container from that image, exposing Streamlit's default port.
resource "docker_container" "boltguard" {
  name  = "boltguard-ai-tf-container"
  image = data.docker_image.boltguard.id

  ports {
    internal = 8501
    external = 8502
  }

  volumes {
    host_path      = "C:/Users/Harini P/boltguard-docker/.streamlit"
    container_path = "/app/.streamlit"
  }
}

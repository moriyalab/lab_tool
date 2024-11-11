#!/bin/bash

# https://qiita.com/k_ikasumipowder/items/5e71208b7c7ae3e4fe7c

ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Prevent running as root.
if [[ $(id -u) -eq 0 ]]; then
    echo "This script cannot be executed with root privileges."
    echo "Please re-run without sudo and follow instructions to configure docker for non-root user if needed."
    exit 1
fi

# Check if user can run docker without root.
RE="\<docker\>"
if [[ ! $(groups $USER) =~ $RE ]]; then
    echo "User |$USER| is not a member of the 'docker' group and cannot run docker commands without sudo."
    echo "Run 'sudo usermod -aG docker \$USER && newgrp docker' to add user to 'docker' group, then re-run this script."
    echo "See: https://docs.docker.com/engine/install/linux-postinstall/"
    exit 1
fi

# Check if able to run docker commands.
if [[ -z "$(docker ps)" ]] ;  then
    echo "Unable to run docker commands. If you have recently added |$USER| to 'docker' group, you may need to log out and log back in for it to take effect."
    echo "Otherwise, please check your Docker installation."
    exit 1
fi

PLATFORM="$(uname -m)"

if [ $PLATFORM = "x86_64" ]; then
    docker pull ghcr.io/moriyalab/lab_tool:latest
    docker run --rm -p 7860:7860 ghcr.io/moriyalab/lab_tool:latest
else
    echo "Not Support Platform. Only support x86."
fi
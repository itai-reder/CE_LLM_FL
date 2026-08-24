#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

log_step() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP] $1"
}

log_info() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1"
}

conda_env_exists() {
	local env_name="$1"
	conda env list | awk 'NR > 2 && $1 !~ /^#/ {print $1}' | grep -Fxq "$env_name"
}

# --- VARIABLES ---
CONDA_ENV_NAME="gzoltar_build"
CONDA_ENV_YML="$PROJECT_ROOT/utils/gzoltar_build.yml"
GZOLTAR_REPO_URL="https://github.com/GZoltar/gzoltar.git"
GZOLTAR_REPO_DIR="$PROJECT_ROOT/utils/gzoltar"
GZOLTAR_VERSION="1.7.4-SNAPSHOT"  # Change this to the desired version or commit hash
GZOLTAR_CLI_SRC_JAR="$GZOLTAR_REPO_DIR/com.gzoltar.cli/target/com.gzoltar.cli-$GZOLTAR_VERSION-jar-with-dependencies.jar"
GZOLTAR_AGENT_SRC_PATH="$GZOLTAR_REPO_DIR/com.gzoltar.agent.rt/target/com.gzoltar.agent.rt-$GZOLTAR_VERSION-all.jar"
GZOLTAR_CLI_DEST_NAME="gzoltar-cli.jar"
GZOLTAR_AGENT_DEST_NAME="gzoltar-agent.jar"
GZOLTAR_DEST_DIR="$PROJECT_ROOT/utils/java"
GZOLTAR_CLI_DEST_PATH="$GZOLTAR_DEST_DIR/$GZOLTAR_CLI_DEST_NAME"
GZOLTAR_AGENT_DEST_PATH="$GZOLTAR_DEST_DIR/$GZOLTAR_AGENT_DEST_NAME"

# --- SETUP --- (if needed)
## 0. Install dependencies (if not already installed)
# # conda, maven, git, etc.
## 1. Build the conda environment (if not already built)
# - Option A: Creation command
# conda create -y -n gzoltar_build -c conda-forge openjdk=8 maven
# - Option B: Using the YML file
# conda env create -f utils/gzoltar_build.yml
## 2. Clone the GZoltar repository (if not already cloned)
# git clone "$GZOLTAR_REPO_URL" $GZOLTAR_REPO_DIR
## 3. Check out to the desired version (if needed)
# cd $GZOLTAR_REPO_DIR && git checkout $GZOLTAR_VERSION # *IF* the version is a valid commit hash
## 4. IF NEEDED: Change variables above to match the actual version/commit hash and paths

# --- BUILD ---

log_step "Starting GZoltar build workflow"
log_info "Project root: $PROJECT_ROOT"
log_info "GZoltar repository: $GZOLTAR_REPO_DIR"
log_info "Destination directory: $GZOLTAR_DEST_DIR"

## 0. Start at the GZoltar repository directory
if [ ! -d "$GZOLTAR_REPO_DIR" ]; then
	log_info "GZoltar repository not found at $GZOLTAR_REPO_DIR"
	# Prompt the user to:
	echo "Please choose an option:"
	# 1) Clone to $GZOLTAR_REPO_DIR (default)
	echo "1) Clone to \`$GZOLTAR_REPO_DIR\` (default)"
	echo "2) Specify a different path"
	echo "3) Exit" # Print instructions to clone manually 
	read -p "Enter your choice (1/2/3) [1]: " choice
	choice=${choice:-1}
	case $choice in
		1)
			log_info "Cloning GZoltar repository to $GZOLTAR_REPO_DIR"
			git clone "$GZOLTAR_REPO_URL" "$GZOLTAR_REPO_DIR"
			cd "$GZOLTAR_REPO_DIR"
			;;
		2)
			read -p "Enter the path to your existing GZoltar repository: " custom_path
			if [ -d "$custom_path" ]; then
				log_info "Using existing GZoltar repository at $custom_path"
				cd "$custom_path"
			else
				log_info "Directory not found at $custom_path"
				echo "Please clone the GZoltar repository manually and run this script again."
				exit 1
			fi
			;;
		3)
			log_info "Exiting. Please clone the GZoltar repository manually and run this script again."
			exit 0
			;;
		*)
			log_info "Invalid choice. Exiting."
			exit 1
			;;
	esac
else
	log_info "GZoltar repository found at $GZOLTAR_REPO_DIR"
fi

log_step "Changing directory to GZoltar repository"
cd "$GZOLTAR_REPO_DIR"
log_info "Current directory: $(pwd)"

## 1. Activate the conda environment
# A. If YML does not exist: Create the environment first
log_step "Preparing conda environment '$CONDA_ENV_NAME'"
if conda_env_exists "$CONDA_ENV_NAME"; then
	log_info "Conda environment '$CONDA_ENV_NAME' already exists"
	if [ -f "$CONDA_ENV_YML" ]; then
		echo "Conda environment file found at $CONDA_ENV_YML"
		echo "Updating environment '$CONDA_ENV_NAME' from YML..."
		conda env update -n "$CONDA_ENV_NAME" -f "$CONDA_ENV_YML" --prune
	else
		echo "Conda environment file not found at $CONDA_ENV_YML"
		echo "Skipping update and using existing environment '$CONDA_ENV_NAME'"
	fi
else
	log_info "Conda environment '$CONDA_ENV_NAME' does not exist"
	if [ -f "$CONDA_ENV_YML" ]; then
		echo "Conda environment file found at $CONDA_ENV_YML"
		echo "Creating environment '$CONDA_ENV_NAME' from YML..."
		conda env create -n "$CONDA_ENV_NAME" -f "$CONDA_ENV_YML"
	else
		echo "Conda environment file not found at $CONDA_ENV_YML"
		echo "Creating environment '$CONDA_ENV_NAME' using conda-forge packages..."
		conda create -y -n "$CONDA_ENV_NAME" -c conda-forge openjdk=8 maven
	fi
fi

# Initialize conda for non-interactive shells and activate environment.
log_step "Activating conda environment"
# Some conda activate scripts read optional vars that are unset under `set -u`.
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"
set -u

echo "Using conda environment: $CONDA_ENV_NAME"
## 2. Build the JAR files using Maven
log_step "Running Maven build"
mvn clean package -DskipTests
log_info "Maven build completed"

## 3. Copy the built JAR files to the destination directory
log_step "Copying built JAR files"
cp "$GZOLTAR_CLI_SRC_JAR" "$GZOLTAR_CLI_DEST_PATH"
log_info "Copied CLI JAR to $GZOLTAR_CLI_DEST_PATH"
cp "$GZOLTAR_AGENT_SRC_PATH" "$GZOLTAR_AGENT_DEST_PATH"
log_info "Copied agent JAR to $GZOLTAR_AGENT_DEST_PATH"

log_step "GZoltar build workflow completed successfully"
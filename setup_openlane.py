import os
import sys
import shutil
import subprocess
import logging

# --- Configuration ---
OPENLANE_VERSION = "version-2.1"
PDK = "gf180mcu"
PDK_ROOT = "~/.volare"

# --- Automatic Path Configuration ---
# All build files are placed in a clean directory in the user's home folder
# to avoid conflicts with existing Git repositories, which is a common
# source of errors with Nix.
HOME_DIR = os.path.expanduser('~')
INSTALL_DIR = os.path.join(HOME_DIR, "openlane_install_files")

def run_command(command, cwd=None, shell=False):
    """
    Runs a command and handles errors, printing the command for clarity.
    """
    cmd_str = command if shell else ' '.join(command)
    print(f"▶️  Running: {cmd_str}")
    try:
        # Using shell=True for commands that involve pipes or sudo redirection.
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
            shell=shell,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except FileNotFoundError:
        tool = command.split()[0] if shell else command[0]
        print(f"❌ Error: Command '{tool}' not found. Please ensure it is installed and in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}.")
        sys.exit(1)

def check_and_install_nix():
    """
    Checks if Nix is installed and, if not, installs it using the official script.
    This function handles the necessary administrative privileges.
    """
    if shutil.which("nix-env"):
        print("✅ Nix is already installed.")
        return

    print("--- ⚙️  Step 1: Installing Nix Package Manager ---")
    print("🔎 Nix not found. Starting installation...")
    print("🔔 This will require administrative privileges (sudo) and will prompt for your password.")

    # Install Nix using the official script. The `| bash` part requires shell=True.
    nix_install_cmd = "curl -L https://nixos.org/nix/install | bash -s -- --daemon --yes"
    run_command(nix_install_cmd, shell=True)

    # Enable the 'flakes' experimental feature by appending to the Nix config file.
    # This requires sudo privileges.
    flakes_config_line = "'extra-experimental-features = nix-command flakes'"
    nix_conf_path = "/etc/nix/nix.conf"
    config_cmd = f"echo {flakes_config_line} | sudo tee -a {nix_conf_path}"
    run_command(config_cmd, shell=True)

    # Restart the Nix daemon to apply the changes.
    kill_daemon_cmd = "sudo killall nix-daemon"
    run_command(kill_daemon_cmd, shell=True, check=False) # check=False because it may not be running

    print("✅ Nix installed and configured successfully.")
    print("🔔 You may need to restart your terminal for all changes to take effect.")
    # Add Nix to the PATH for the current session to ensure subsequent commands work.
    nix_profile_path = "/nix/var/nix/profiles/default/bin/"
    if nix_profile_path not in os.environ["PATH"]:
        os.environ["PATH"] = f"{nix_profile_path}:{os.getenv('PATH')}"


def setup_openlane_source():
    """
    Downloads and extracts the OpenLane source code into a clean directory.
    """
    print(f"--- ⬇️  Step 2: Downloading OpenLane source to '{INSTALL_DIR}' ---")
    if os.path.exists(INSTALL_DIR):
        print(f"🗑️ Removing existing installation directory...")
        shutil.rmtree(INSTALL_DIR)
    os.makedirs(INSTALL_DIR)

    version = "main" if OPENLANE_VERSION == "latest" else OPENLANE_VERSION
    url = f"https://github.com/efabless/openlane2/tarball/{version}"

    # Use a direct pipe for efficiency and to avoid intermediate files.
    download_command = f'curl -L "{url}" | tar -xzC {INSTALL_DIR} --strip-components 1'
    run_command(download_command, shell=True)
    print("✅ OpenLane source downloaded successfully.")

def install_dependencies():
    """
    Installs both Nix and Python dependencies for OpenLane.
    """
    # Install Nix dependencies
    print("\n--- 📦 Step 3: Installing Nix Dependencies ---")
    nix_command = "nix profile install .#colab-env --accept-flake-config"
    run_command(nix_command, cwd=INSTALL_DIR, shell=True)
    print("✅ Nix dependencies installed.")

    # Install Python dependencies
    print("\n--- 🐍 Step 4: Installing Python Dependencies ---")
    # Use sys.executable to ensure we use the correct pip for the current Python env.
    pip_command = f'"{sys.executable}" -m pip install .'
    run_command(pip_command, cwd=INSTALL_DIR, shell=True)
    print("✅ Python dependencies installed.")

def setup_volare_and_pdk():
    """
    Installs Volare and enables the specified PDK.
    """
    print(f"\n--- 🛠️  Step 5: Setting up Volare and enabling PDK '{PDK}' ---")
    
    # Temporarily add the OpenLane installation to Python's path
    # to import the 'volare' module that was just installed.
    sys.path.insert(0, INSTALL_DIR)
    try:
        import volare
        pdk_root_expanded = os.path.expanduser(PDK_ROOT)
        open_pdks_rev_path = os.path.join(INSTALL_DIR, "openlane", "open_pdks_rev")

        with open(open_pdks_rev_path, "r", encoding="utf8") as f:
            open_pdks_rev = f.read().strip()
        
        print(f"Enabling PDK with Open PDKs revision '{open_pdks_rev}'...")
        volare.enable(volare.get_volare_home(pdk_root_expanded), PDK, open_pdks_rev)
        print("✅ PDK enabled successfully.")

    except ImportError:
        print("❌ Critical Error: Failed to import 'volare' after installation.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ An error occurred during PDK setup: {e}")
        sys.exit(1)
    finally:
        # Clean up the path
        if INSTALL_DIR in sys.path:
            sys.path.remove(INSTALL_DIR)

def main():
    """
    Runs the entire setup process.
    """
    print("--- 🚀 Starting OpenLane 2 Local Setup ---")
    
    check_and_install_nix()
    setup_openlane_source()
    install_dependencies()
    setup_volare_and_pdk()

    # Final verification to confirm everything is working
    print("\n--- Verifying installation ---")
    sys.path.insert(0, INSTALL_DIR)
    try:
        import openlane
        print(f"✅ Success! OpenLane version {openlane.__version__} is installed.")
    except ImportError:
        print("❌ Verification failed. Could not import OpenLane.")
        sys.exit(1)
    finally:
        if INSTALL_DIR in sys.path:
            sys.path.remove(INSTALL_DIR)

    # Clear any default loggers to prevent conflicts
    logging.getLogger().handlers.clear()

    print("\n\n🎉 OpenLane setup is complete!")
    print(f"   Installation Location: {INSTALL_DIR}")
    print("   You may need to restart your terminal for all environment changes to take effect.")

if __name__ == "__main__":
    main()
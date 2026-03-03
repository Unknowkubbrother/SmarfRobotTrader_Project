#!/bin/bash

# Configuration variables
mt5file='/config/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe'
WINEPREFIX='/config/.wine'
WINEDEBUG='-all'
export WINEARCH=win64
export WINEDLLOVERRIDES="mscoree,mshtml="
wine_executable="/usr/lib/wine/wine64"
metatrader_version="5.0.36"
mt5server_port="8001"
PYTHON_DIR="/config/.wine/drive_c/Python311"
WINE_BOOTSTRAP_MARKER="/config/.wine/.wine_python_bootstrap_v1"
MT5_SNAPSHOT_PATH="${MT5_SNAPSHOT_PATH:-}"

# Installer paths (pre-downloaded in Dockerfile)
mono_installer="/defaults/installers/wine-mono-10.3.0-x86.msi"
python_installer="/defaults/installers/python-3.8.10.exe"
mt5_installer="/defaults/installers/mt5setup.exe"

# Fallback URLs
mono_url="https://dl.winehq.org/wine/wine-mono/10.3.0/wine-mono-10.3.0-x86.msi"
python_url="https://www.python.org/ftp/python/3.8.10/python-3.8.10.exe"
mt5setup_url="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"

# Setup logging
mkdir -p /config
exec > >(tee -a /config/startup.log) 2>&1
echo "----------------------------------------------------------------"
echo "Starting MT5 startup script at $(date)"
echo "----------------------------------------------------------------"

# Function to display a graphical message
show_message() {
    echo $1
}

# Function to check if a dependency is installed
check_dependency() {
    if ! command -v $1 &> /dev/null; then
        echo "$1 is not installed. Please install it to continue."
        exit 1
    fi
}

# Function to check if a Python package is installed
is_python_package_installed() {
    python3 -c "import pkg_resources; exit(not pkg_resources.require('$1'))" 2>/dev/null
    return $?
}

# Function to check if a Python package is installed in Wine
is_wine_python_package_installed() {
    $wine_executable python -c "import pkg_resources; exit(not pkg_resources.require('$1'))" 2>/dev/null
    return $?
}

# Check for necessary dependencies
check_dependency "curl"
check_dependency "$wine_executable"

# Optional: restore a prebuilt MT5/Wine snapshot to skip first-time installation.
if [ ! -e "$mt5file" ]; then
    snapshot_candidates=()
    if [ -n "$MT5_SNAPSHOT_PATH" ]; then
        snapshot_candidates+=("$MT5_SNAPSHOT_PATH")
    fi
    snapshot_candidates+=(/bots/mt5-config-snapshot.tgz /bots/mt5-config-snapshot.tar.gz /config/mt5-config-snapshot.tgz /config/mt5-config-snapshot.tar.gz)

    for snapshot in "${snapshot_candidates[@]}"; do
        if [ -f "$snapshot" ]; then
            show_message "[0/7] Restoring MT5 snapshot from $snapshot ..."
            mkdir -p /config
            if tar -xzf "$snapshot" -C /config; then
                if [ -e "$mt5file" ]; then
                    show_message "[0/7] Snapshot restored successfully."
                    break
                fi
                show_message "[0/7] Snapshot restored but MT5 not found, fallback to installer."
            else
                show_message "[0/7] Snapshot restore failed, fallback to installer."
            fi
        fi
    done
fi

# Install Mono if not present
if [ ! -e "/config/.wine/drive_c/windows/mono" ]; then
    show_message "[1/7] Installing Mono..."
    if [ -f "$mono_installer" ]; then
        cp "$mono_installer" /tmp/mono.msi
    else
        show_message "Local Mono installer not found, downloading..."
        curl -o /tmp/mono.msi $mono_url
    fi
    WINEDLLOVERRIDES=mscoree=d $wine_executable msiexec /i /tmp/mono.msi /qn
    rm /tmp/mono.msi
    show_message "[1/7] Mono installed."
else
    show_message "[1/7] Mono is already installed."
fi

# Check if MetaTrader 5 is already installed
if [ -e "$mt5file" ]; then
    show_message "[2/7] File $mt5file already exists."
else
    show_message "[2/7] File $mt5file is not installed. Installing..."

    # Set Windows 10 mode in Wine and download and install MT5
    $wine_executable wineboot --init
    $wine_executable reg add "HKEY_CURRENT_USER\\Software\\Wine" /v Version /t REG_SZ /d "win10" /f
    show_message "[3/7] Preparing MT5 installer..."
    if [ -f "$mt5_installer" ]; then
        cp "$mt5_installer" /tmp/mt5setup.exe
    else
        show_message "Local MT5 installer not found, downloading..."
        curl -o /tmp/mt5setup.exe $mt5setup_url
    fi
    show_message "[3/7] Installing MetaTrader 5..."
    show_message "Running MT5 installer with /auto flag..."
    
    # Start installer in background
    $wine_executable "/tmp/mt5setup.exe" "/auto" &
    INSTALLER_PID=$!
    
    # Wait for terminal64.exe to appear
    count=0
    while [ ! -f "$mt5file" ]; do
        sleep 1
        count=$((count+1))
        if [ $count -gt 300 ]; then # 5 minutes timeout
            show_message "Error: Installation timed out."
            kill $INSTALLER_PID
            exit 1
        fi
        # Check if installer died prematurely
        if ! kill -0 $INSTALLER_PID 2>/dev/null; then
             show_message "Installer exited"
             break
        fi
    done
    
    if [ -f "$mt5file" ]; then
        show_message "Installation detected. Waiting for files to settle..."
        sleep 10
        
        # Kill the installer if it's still running (stuck on 'Finish' screen)
        if kill -0 $INSTALLER_PID 2>/dev/null; then
            show_message "Closing installer process..."
            kill $INSTALLER_PID
            wait $INSTALLER_PID 2>/dev/null
        fi
    fi

    rm -f /tmp/mt5setup.exe
fi

# Recheck if MetaTrader 5 is installed
if [ -e "$mt5file" ]; then
    show_message "[4/7] File $mt5file is installed. Running MT5..."
    # Change directory to MT5 folder to ensure it finds its resources
    cd "/config/.wine/drive_c/Program Files/MetaTrader 5/"
    $wine_executable terminal64.exe &
else
    show_message "[4/7] File $mt5file is not installed. MT5 cannot be run."
fi


# Ensure Wine is set to Windows 10 mode for Python 3.9+ compatibility
$wine_executable reg add "HKEY_CURRENT_USER\\Software\\Wine" /v Version /t REG_SZ /d "win10" /f

# Install Python in Wine (Embeddable Package method for reliability)
# We use the 64-bit version to match Wine64
PYTHON_ZIP="/defaults/installers/python-3.11.9-embed-amd64.zip"

if [ ! -d "$PYTHON_DIR" ]; then
    show_message "[5/7] Installing Python 3.11.9 (Embeddable) in Wine..."
    
    # Create directory
    mkdir -p "$PYTHON_DIR"
    
    # Extract (using unzip from host)
    if [ -f "$PYTHON_ZIP" ]; then
        unzip -q "$PYTHON_ZIP" -d "$PYTHON_DIR"
    else
        show_message "Python zip not found, downloading..."
        curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip" -o /tmp/python.zip
        unzip -q /tmp/python.zip -d "$PYTHON_DIR"
        rm /tmp/python.zip
    fi
    
    # Enable site-packages (required for pip)
    # The embeddable package has a ._pth file that restricts imports. We need to uncomment 'import site'.
    sed -i 's/^#import site/import site/' "$PYTHON_DIR/python311._pth"
    
    # Download get-pip.py
    show_message "Downloading get-pip.py..."
    rm -f "$PYTHON_DIR/get-pip.py"
    curl -L https://bootstrap.pypa.io/get-pip.py -o "$PYTHON_DIR/get-pip.py"
    
    # Verify Python version
    show_message "Verifying Python version..."
    $wine_executable "$PYTHON_DIR/python.exe" --version
    
    # Install pip
    show_message "Installing pip..."
    $wine_executable "$PYTHON_DIR/python.exe" "$PYTHON_DIR/get-pip.py" --no-warn-script-location || show_message "Pip installation failed!"
    
    # Add to PATH (registry)
    show_message "Updating Windows PATH..."
    $wine_executable reg add "HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control\\Session Manager\\Environment" /v Path /t REG_EXPAND_SZ /d "C:\\windows\\system32;C:\\windows;C:\\windows\\system32\\wbem;C:\\Python311;C:\\Python311\\Scripts" /f
    
    show_message "[5/7] Python installed in Wine."
else
    show_message "[5/7] Python is already installed in Wine."
fi

# Upgrade pip and install required packages (once per config volume)
WINE_PYTHON="$PYTHON_DIR/python.exe"
if [ -x "$WINE_PYTHON" ] && [ -f "$WINE_BOOTSTRAP_MARKER" ]; then
    show_message "[6/7] Python libraries already installed in Wine. Skipping."
else
    show_message "[6/7] Installing Python libraries"
    $wine_executable "$WINE_PYTHON" -m pip install --upgrade pip
    $wine_executable "$WINE_PYTHON" -m pip install --no-warn-script-location "MetaTrader5<5.0.5500"
    # Pin rpyc to 5.0.1 to match Linux side for protocol compatibility
    # Pin numpy to 1.24.4 - last version that works in Wine without ucrtbase.dll.crealf
    # Install mt5linux with --no-deps to avoid building cffi/cryptography from source
    $wine_executable "$WINE_PYTHON" -m pip install --no-warn-script-location --no-deps mt5linux rpyc==5.0.1 plumbum "numpy<1.25" pyzmq
    $wine_executable "$WINE_PYTHON" -m pip install --no-cache-dir python-dateutil
    touch "$WINE_BOOTSTRAP_MARKER"
    show_message "[6/7] Python libraries installed in Wine."
fi

# Backward-compatible safety check for old volumes without marker.
if ! is_wine_python_package_installed "python-dateutil"; then
    show_message "[6/7] Installing missing python-dateutil library in Windows"
    $wine_executable "$WINE_PYTHON" -m pip install --no-cache-dir python-dateutil
    touch "$WINE_BOOTSTRAP_MARKER"
fi

# Install mt5linux library in Linux if not installed.
show_message "[6/7] Checking and installing mt5linux library in Linux if necessary"
if ! is_python_package_installed "mt5linux"; then
    pip install --break-system-packages --no-cache-dir --no-deps mt5linux && \
    pip install --break-system-packages --no-cache-dir rpyc==5.0.1 plumbum numpy
fi

# Install pyxdg library in Linux if not installed
show_message "[6/7] Checking and installing pyxdg library in Linux if necessary"
if ! is_python_package_installed "pyxdg"; then
    pip install --break-system-packages --no-cache-dir pyxdg
fi

# Start the MT5 server on Linux
show_message "[7/7] Starting the mt5linux server..."
# Run server inside Wine Python directly (compatible with mt5linux>=0.2.4).
$wine_executable "$WINE_PYTHON" -m mt5linux --host 0.0.0.0 -p $mt5server_port &

# Wait up to 60s for server socket to appear
server_ready=0
for _ in $(seq 1 30); do
    if ss -tuln | grep ":$mt5server_port" > /dev/null; then
        server_ready=1
        break
    fi
    sleep 2
done

if [ "$server_ready" -eq 1 ]; then
    show_message "[7/7] The mt5linux server is running on port $mt5server_port."
else
    show_message "[7/7] Failed to start the mt5linux server on port $mt5server_port."
fi

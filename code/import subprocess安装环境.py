import subprocess
import sys

def install_packages():
    """安装必要的包"""
    try:
        # 首先检查是否需要升级pip
        result = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Warning: Failed to upgrade pip: {result.stderr}")
        
        # 安装所需的包
        packages = ["skl2onnx", "onnx", "onnxruntime"]
        for package in packages:
            print(f"Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"Successfully installed {package}")
            else:
                print(f"Failed to install {package}: {result.stderr}")
                return False
        
        print("All packages installed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        print("Error: Installation timed out")
        return False
    except Exception as e:
        print(f"Error during installation: {str(e)}")
        return False

if __name__ == "__main__":
    install_packages()

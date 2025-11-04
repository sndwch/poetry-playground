"""System utilities for checking dependencies."""

import subprocess
import sys
import platform


def check_poppler_installed():
    """Check if Poppler is installed and available in PATH."""
    try:
        # Try to run pdfinfo which is part of poppler-utils
        result = subprocess.run(['pdfinfo', '-v'],
                              capture_output=True,
                              text=True,
                              timeout=2)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def get_poppler_install_instructions():
    """Get platform-specific installation instructions for Poppler."""
    system = platform.system()

    if system == "Darwin":  # macOS
        return "Install with: brew install poppler"
    elif system == "Linux":
        # Try to detect Linux distribution
        try:
            with open('/etc/os-release', 'r') as f:
                content = f.read().lower()
                if 'ubuntu' in content or 'debian' in content:
                    return "Install with: sudo apt-get install poppler-utils"
                elif 'fedora' in content or 'rhel' in content or 'centos' in content:
                    return "Install with: sudo yum install poppler-utils"
                else:
                    return "Install poppler-utils using your distribution's package manager"
        except:
            return "Install poppler-utils using your distribution's package manager"
    elif system == "Windows":
        return "Download from: https://github.com/oschwartz10612/poppler-windows/releases/"
    else:
        return "Please install Poppler for your operating system"


def check_hunspell_installed():
    """Check if hunspell is available."""
    try:
        import hunspell
        return True
    except ImportError:
        return False


def check_system_dependencies():
    """Check all system dependencies and print status."""
    print("\nSystem Dependencies Check:")
    print("-" * 40)

    # Check Poppler
    poppler_installed = check_poppler_installed()
    if poppler_installed:
        print("✓ Poppler: Installed (PDF to PNG conversion enabled)")
    else:
        print("✗ Poppler: Not installed (PDF generation will work, PNG conversion disabled)")
        print(f"  {get_poppler_install_instructions()}")

    # Check Hunspell
    hunspell_installed = check_hunspell_installed()
    if hunspell_installed:
        print("✓ Hunspell: Installed (spellchecking enabled)")
    else:
        print("✗ Hunspell: Not installed (spellchecking disabled)")
        print("  Install with: pip install hunspell (requires system hunspell libraries)")

    print("-" * 40)
    return poppler_installed, hunspell_installed


if __name__ == "__main__":
    check_system_dependencies()
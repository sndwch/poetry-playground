"""System utilities for checking dependencies."""

import platform
import subprocess


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
            with open('/etc/os-release') as f:
                content = f.read().lower()
                if 'ubuntu' in content or 'debian' in content:
                    return "Install with: sudo apt-get install poppler-utils"
                elif 'fedora' in content or 'rhel' in content or 'centos' in content:
                    return "Install with: sudo yum install poppler-utils"
                else:
                    return "Install poppler-utils using your distribution's package manager"
        except OSError:
            return "Install poppler-utils using your distribution's package manager"
    elif system == "Windows":
        return "Download from: https://github.com/oschwartz10612/poppler-windows/releases/"
    else:
        return "Please install Poppler for your operating system"


def check_hunspell_installed():
    """Check if hunspell Python package and system libraries are available."""
    try:
        import hunspell
        # Try to actually initialize it to verify system libraries
        system = platform.system()
        if system == "Darwin":
            # Try macOS default location
            try:
                hunspell.HunSpell('/Library/Spelling/en_US.dic', '/Library/Spelling/en_US.aff')
                return True
            except Exception:
                return False
        elif system == "Linux":
            # Try Linux default location
            try:
                hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
                return True
            except Exception:
                return False
        else:
            # Windows or other OS
            return False
    except ImportError:
        return False


def get_hunspell_install_instructions():
    """Get platform-specific installation instructions for Hunspell."""
    system = platform.system()

    if system == "Darwin":  # macOS
        return """Install with:
  1. brew install hunspell
  2. pip install hunspell
  Note: You may need system dictionaries in /Library/Spelling/"""
    elif system == "Linux":
        # Try to detect Linux distribution
        try:
            with open('/etc/os-release') as f:
                content = f.read().lower()
                if 'ubuntu' in content or 'debian' in content:
                    return """Install with:
  1. sudo apt-get install libhunspell-dev hunspell-en-us
  2. pip install hunspell"""
                elif 'fedora' in content or 'rhel' in content or 'centos' in content:
                    return """Install with:
  1. sudo yum install hunspell-devel hunspell-en
  2. pip install hunspell"""
                else:
                    return """Install hunspell using your distribution's package manager:
  1. Install hunspell development libraries and dictionaries
  2. pip install hunspell"""
        except OSError:
            return """Install hunspell using your distribution's package manager:
  1. Install hunspell development libraries and dictionaries
  2. pip install hunspell"""
    elif system == "Windows":
        return """Hunspell setup on Windows requires manual configuration:
  1. Download hunspell binaries from: https://github.com/hunspell/hunspell/releases
  2. Install Python package: pip install hunspell
  3. Configure dictionary paths
  Note: This is optional - spellchecking features will be disabled without it"""
    else:
        return "Please install Hunspell for your operating system (optional)"


def check_system_dependencies():
    """Check all system dependencies and print detailed status with installation instructions.

    Returns:
        tuple: (poppler_installed: bool, hunspell_installed: bool)
    """
    print("\n" + "=" * 70)
    print("System Dependencies Check")
    print("=" * 70)

    all_ok = True

    # Check Poppler
    print("\n📄 PDF to PNG Conversion (Poppler):")
    poppler_installed = check_poppler_installed()
    if poppler_installed:
        print("  ✓ Status: Installed and working")
        print("  ✓ PDF to PNG conversion: ENABLED")
    else:
        all_ok = False
        print("  ✗ Status: Not found")
        print("  ✗ PDF to PNG conversion: DISABLED")
        print("\n  Installation instructions:")
        instructions = get_poppler_install_instructions()
        for line in instructions.split('\n'):
            print(f"  {line}")
        print("\n  Note: This is OPTIONAL. You can still generate PDFs without it.")

    # Check Hunspell
    print("\n📝 Spellchecking (Hunspell):")
    hunspell_installed = check_hunspell_installed()
    if hunspell_installed:
        print("  ✓ Status: Installed and working")
        print("  ✓ Spellchecking features: ENABLED")
    else:
        all_ok = False
        print("  ✗ Status: Not found")
        print("  ✗ Spellchecking features: DISABLED")
        print("\n  Installation instructions:")
        instructions = get_hunspell_install_instructions()
        for line in instructions.split('\n'):
            print(f"  {line}")
        print("\n  Note: This is OPTIONAL. Most features work without it.")

    # Summary
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All optional dependencies are installed!")
    else:
        print("ℹ  Some optional dependencies are missing (see above for instructions)")
        print("ℹ  Core features will work - these are optional enhancements")
    print("=" * 70 + "\n")

    return poppler_installed, hunspell_installed


if __name__ == "__main__":
    check_system_dependencies()

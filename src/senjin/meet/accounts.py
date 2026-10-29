"""Managing the bot's saved Google sign-in (the Chrome profile in data/meet_profile)."""

import shutil
import subprocess
import time
from pathlib import Path

from senjin.meet.joiner import PROFILE_DIR


class ProfileError(RuntimeError):
    pass


def profile_in_use(profile_dir: Path) -> bool:
    """True while a Chrome started with this profile is running."""
    result = subprocess.run(
        ["pgrep", "-f", f"user-data-dir={profile_dir.resolve()}"], capture_output=True
    )
    return result.returncode == 0


def _check_free(profile_dir: Path, in_use) -> None:
    if in_use(profile_dir):
        raise ProfileError("the bot's Chrome is still running; stop senjin-meet first")


def logout(profile_dir: Path = PROFILE_DIR, in_use=profile_in_use) -> bool:
    """Delete the saved sign-in. Returns False if there was nothing to delete."""
    if not profile_dir.exists():
        return False
    _check_free(profile_dir, in_use)
    shutil.rmtree(profile_dir)
    return True


def keep_account(
    profile_dir: Path = PROFILE_DIR, label: str | None = None, in_use=profile_in_use
) -> Path | None:
    """Set the current sign-in aside (not deleted) so another account can be signed in.

    Returns where it was moved, or None if no profile existed. To switch back, run logout and
    move the saved folder back to data/meet_profile.
    """
    if not profile_dir.exists():
        return None
    _check_free(profile_dir, in_use)
    dest = profile_dir.with_name(f"{profile_dir.name}_{label or time.strftime('%Y%m%d-%H%M%S')}")
    if dest.exists():
        raise ProfileError(f"{dest} already exists; choose another --label")
    profile_dir.rename(dest)
    return dest

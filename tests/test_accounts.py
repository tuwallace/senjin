import pytest

from senjin.meet.accounts import ProfileError, keep_account, logout


def make_profile(tmp_path):
    profile = tmp_path / "meet_profile"
    (profile / "Default").mkdir(parents=True)
    (profile / "Default" / "Cookies").write_text("x")
    return profile


def test_logout_deletes_the_profile(tmp_path):
    profile = make_profile(tmp_path)
    assert logout(profile, in_use=lambda p: False) is True
    assert not profile.exists()
    assert logout(profile, in_use=lambda p: False) is False  # already signed out


def test_logout_refuses_while_chrome_is_running(tmp_path):
    profile = make_profile(tmp_path)
    with pytest.raises(ProfileError, match="still running"):
        logout(profile, in_use=lambda p: True)
    assert profile.exists()


def test_keep_account_moves_not_deletes(tmp_path):
    profile = make_profile(tmp_path)
    saved = keep_account(profile, "work", in_use=lambda p: False)
    assert saved == tmp_path / "meet_profile_work"
    assert (saved / "Default" / "Cookies").read_text() == "x"
    assert not profile.exists()


def test_keep_account_refuses_to_overwrite_and_handles_no_profile(tmp_path):
    profile = make_profile(tmp_path)
    (tmp_path / "meet_profile_work").mkdir()
    with pytest.raises(ProfileError, match="already exists"):
        keep_account(profile, "work", in_use=lambda p: False)
    assert profile.exists()
    assert keep_account(tmp_path / "none", in_use=lambda p: False) is None

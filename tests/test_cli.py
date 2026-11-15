"""CLI wiring without weights."""
import pytest

from hwdet import __version__, cli


def test_version(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_models_list(capsys):
    cli.main(["models", "list"])
    assert "hwdet-s@1.0.0" in capsys.readouterr().out


def test_data_manifest(tmp_path, capsys):
    (tmp_path / "train" / "images").mkdir(parents=True)
    cli.main(["data", "manifest", str(tmp_path)])
    assert '"sha256"' in capsys.readouterr().out

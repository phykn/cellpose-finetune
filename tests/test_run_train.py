from pathlib import Path

from run_train import make_run_dir, parse_args


def test_parse_args_accepts_explicit_paths(tmp_path) -> None:
    config = tmp_path / "train.yaml"
    run_dir = tmp_path / "run"

    args = parse_args(
        [
            "--config",
            str(config),
            "--device",
            "cpu",
            "--run-dir",
            str(run_dir),
        ]
    )

    assert args.config == config
    assert args.device == "cpu"
    assert args.run_dir == run_dir


def test_make_run_dir_avoids_existing_name(monkeypatch, tmp_path) -> None:
    class FixedDateTime:
        @classmethod
        def now(cls):
            return cls()

        def astimezone(self):
            return self

        def strftime(self, pattern: str) -> str:
            assert pattern == "%m%d%H%M"
            return "08091234"

    monkeypatch.setattr("run_train.datetime", FixedDateTime)
    (tmp_path / "08091234").mkdir()

    path = make_run_dir(Path(tmp_path))

    assert path == tmp_path / "0809123402"
